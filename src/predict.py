"""Prediction module: score how well each resume fits a job description.

Loads the final XGBoost hybrid model once and, for one JD and several resumes, returns
the fit label, confidence, fit score, and matched / missing skills, ranked best first.

Model artifacts (models/) are plain data, so they do not depend on the scikit-learn
version used in Colab:
  xgb_hybrid.json        XGBoost model (JSON format)
  xgb_hybrid_meta.json   feature order and class order
  pca_32.npz             PCA mean + 32 components (embedding comparisons -> 32 values)
  tfidf.json             TF-IDF vocabulary + idf weights (fitted on train texts only)
  skills.csv             203-skill list with variants

Feature logic is copied exactly from notebook 03 (see src/preprocess.py).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier

from src.preprocess import build_skill_patterns, clean_text, extract_skills, jd_years, resume_years

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK = 180   # words per chunk, as in notebook 03 (within the model's 256-token limit)


class Matcher:
    def __init__(self, model_dir=MODEL_DIR, device="cpu"):
        model_dir = Path(model_dir)

        meta = json.loads((model_dir / "xgb_hybrid_meta.json").read_text())
        self.features = meta["features"]
        self.order = meta["order"]

        self.xgb = XGBClassifier()
        self.xgb.load_model(str(model_dir / "xgb_hybrid.json"))

        pca = np.load(model_dir / "pca_32.npz")
        self.pca_mean, self.pca_components = pca["mean"], pca["components"]

        # TF-IDF rebuilt from plain data: counts -> 1 + log(tf) -> x idf -> L2 normalise
        tf = json.loads((model_dir / "tfidf.json").read_text())
        self.tfidf_counter = CountVectorizer(ngram_range=tuple(tf["ngram_range"]), stop_words="english",
                                             vocabulary=tf["vocabulary"])
        self.idf = np.asarray(tf["idf"])
        self.sublinear_tf = tf["sublinear_tf"]

        self.skill_patterns = build_skill_patterns(pd.read_csv(model_dir / "skills.csv"))

        from sentence_transformers import SentenceTransformer
        self.sbert = SentenceTransformer(EMBED_MODEL, device=device)

    # ---------------- building blocks ----------------
    def tfidf(self, texts):
        X = self.tfidf_counter.transform(texts).astype(np.float64).tocsr()
        if self.sublinear_tf:
            X.data = np.log(X.data) + 1.0
        X = csr_matrix(X.multiply(self.idf))
        return normalize(X, norm="l2")

    def embed(self, texts):
        """Split each text into 180-word chunks, embed, sum and L2-normalise (notebook 03)."""
        all_chunks, owner = [], []
        for i, t in enumerate(texts):
            w = t.split()
            for c in ([" ".join(w[j:j + CHUNK]) for j in range(0, len(w), CHUNK)] or [""]):
                all_chunks.append(c); owner.append(i)
        E = self.sbert.encode(all_chunks, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        doc = np.zeros((len(texts), E.shape[1]), dtype=np.float32)
        np.add.at(doc, np.array(owner), E)
        doc /= np.linalg.norm(doc, axis=1, keepdims=True)
        return doc

    def pair_features(self, jds, resumes, emb_jd=None, emb_res=None):
        """14 engineered features + 768 embedding comparisons for each (jd, resume) pair."""
        rows, details = [], []
        for jd, res in zip(jds, resumes):
            j_sk, r_sk = extract_skills(jd, self.skill_patterns), extract_skills(res, self.skill_patterns)
            matched, missing = j_sk & r_sk, j_sk - r_sk
            jy, ry = jd_years(jd), resume_years(res)
            gap = ry - jy
            rows.append({
                "skill_coverage": len(matched) / len(j_sk) if j_sk else 0.0,
                "skill_jaccard": len(j_sk & r_sk) / len(j_sk | r_sk) if (j_sk | r_sk) else 0.0,
                "matched_count": len(matched), "missing_count": len(missing),
                "jd_skill_count": len(j_sk), "resume_skill_count": len(r_sk),
                "jd_years": jy, "resume_years": ry, "exp_gap": gap,
                "exp_meets": np.nan if np.isnan(gap) else float(gap >= 0),
                "jd_years_missing": int(np.isnan(jy)), "resume_years_missing": int(np.isnan(ry)),
            })
            details.append({"matched_skills": sorted(matched), "missing_skills": sorted(missing)})

        F = pd.DataFrame(rows)
        J, R = self.tfidf(list(jds)), self.tfidf(list(resumes))
        F["tfidf_cosine"] = np.asarray(J.multiply(R).sum(axis=1)).ravel()

        if emb_jd is None:
            emb_jd, emb_res = self.embed(list(jds)), self.embed(list(resumes))
        F["embed_cosine"] = (emb_jd * emb_res).sum(axis=1)
        Xe = np.hstack([np.abs(emb_jd - emb_res), emb_jd * emb_res])
        return F[self.features], Xe, details

    def predict_proba(self, F, Xe):
        P = (Xe - self.pca_mean) @ self.pca_components.T          # PCA transform (no whitening)
        X = np.hstack([F.values.astype(np.float64), P])
        return self.xgb.predict_proba(X)                           # columns follow self.order

    # ---------------- main entry point ----------------
    def match(self, jd_text, resumes):
        """jd_text: raw job description. resumes: {name: raw resume text}.
        Returns a list of results ranked by fit score (best first)."""
        t0 = time.time()
        names = list(resumes)
        jd = clean_text(jd_text)
        res = [clean_text(resumes[n]) for n in names]

        emb_jd = np.repeat(self.embed([jd]), len(res), axis=0)
        emb_res = self.embed(res)
        F, Xe, details = self.pair_features([jd] * len(res), res, emb_jd, emb_res)
        P = self.predict_proba(F, Xe)

        results = []
        for i, name in enumerate(names):
            p = P[i]
            results.append({
                "name": name,
                "label": self.order[int(p.argmax())],
                "confidence": round(float(p.max()), 3),
                "probabilities": {c: round(float(v), 3) for c, v in zip(self.order, p)},
                # expected fit (No 0, Potential 1, Good 2), scaled to 0-100; same score used for ranking in notebook 06
                "fit_score": round(float(50 * (p[1] + 2 * p[2])), 1),
                "skill_coverage": round(float(F["skill_coverage"].iloc[i]), 3),
                "semantic_similarity": round(float(F["embed_cosine"].iloc[i]), 3),
                "jd_years_required": None if np.isnan(F["jd_years"].iloc[i]) else float(F["jd_years"].iloc[i]),
                "resume_years": None if np.isnan(F["resume_years"].iloc[i]) else float(F["resume_years"].iloc[i]),
                **details[i],
            })
        results.sort(key=lambda r: r["fit_score"], reverse=True)
        for rank, r in enumerate(results, 1):
            r["rank"] = rank
        elapsed_ms = (time.time() - t0) * 1000
        return {"results": results, "resumes_scored": len(results), "elapsed_ms": round(elapsed_ms, 1)}


if __name__ == "__main__":
    m = Matcher()
    jd = "We need a Python developer with 3+ years of experience in SQL, AWS and Docker."
    out = m.match(jd, {
        "Candidate A": "Software engineer. Skills: Python, SQL, Docker, AWS, Git. Developer, 01/2018 - Current.",
        "Candidate B": "Sales Associate with customer service and retail experience. 2015 - 2019.",
    })
    for r in out["results"]:
        print(r["rank"], r["name"], r["label"], r["confidence"], r["fit_score"], "| missing:", r["missing_skills"])
    print("Time:", out["elapsed_ms"], "ms")
