"""Consistency check: does the laptop pipeline reproduce the Colab results?

Uses data/test_feat.parquet (features computed in Colab) and data/embeddings.joblib
(embeddings computed in Colab). For every test row it compares:
  1. the 14 engineered features recomputed locally from the text vs the Colab values
  2. local Sentence-BERT embeddings vs Colab embeddings
  3. model probabilities from the full local pipeline vs probabilities from Colab's own features

Run from the project root:   python -m src.verify
Optional: python -m src.verify --n 200   (faster, uses a sample of test rows)
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.predict import ROOT, Matcher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="number of test rows to check (0 = all)")
    n = ap.parse_args().n

    test = pd.read_parquet(ROOT / "data" / "test_feat.parquet")
    if n:
        test = test.sample(n, random_state=42).reset_index(drop=True)
    E = joblib.load(ROOT / "data" / "embeddings.joblib")
    colab_emb = dict(zip(E["texts"], E["embeddings"]))
    print(f"Checking {len(test)} test rows ({test['jd'].nunique()} JDs, {test['resume'].nunique()} resumes)")

    m = Matcher()

    # Texts in test_feat are already cleaned, so they are used as-is (no second clean_text pass)
    texts = pd.unique(pd.concat([test["jd"], test["resume"]]))
    local_emb = dict(zip(texts, m.embed(list(texts))))
    emb_cos = np.array([float(local_emb[t] @ colab_emb[t]) for t in texts])
    print(f"\n[Embeddings] cosine(local, Colab): min {emb_cos.min():.6f} | mean {emb_cos.mean():.6f}")

    EJ = np.stack(test["jd"].map(local_emb).values)
    ER = np.stack(test["resume"].map(local_emb).values)
    F_local, Xe_local, _ = m.pair_features(list(test["jd"]), list(test["resume"]), EJ, ER)

    print("\n[Features] max absolute difference, local vs Colab:")
    worst = 0.0
    for c in m.features:
        a, b = F_local[c].values.astype(float), test[c].values.astype(float)
        nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
        both = ~np.isnan(a) & ~np.isnan(b)
        diff = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
        worst = max(worst, diff)
        flag = "" if diff < 1e-3 and nan_mismatch == 0 else "   <-- CHECK"
        print(f"  {c:22s} {diff:.6f}   missing-value mismatches: {nan_mismatch}{flag}")

    # Reference = Colab features + Colab embeddings through the same saved model (what notebook 06 scored)
    CJ = np.stack(test["jd"].map(colab_emb).values)
    CR = np.stack(test["resume"].map(colab_emb).values)
    P_ref = m.predict_proba(test[m.features], np.hstack([np.abs(CJ - CR), CJ * CR]))
    P_loc = m.predict_proba(F_local, Xe_local)

    pred_ref = np.array(m.order)[P_ref.argmax(1)]
    pred_loc = np.array(m.order)[P_loc.argmax(1)]
    print(f"\n[Predictions] max probability difference: {np.abs(P_ref - P_loc).max():.4f}")
    print(f"  same predicted label: {(pred_ref == pred_loc).mean():.1%}")
    print(f"  macro F1 — reference (Colab features): {f1_score(test['label'], pred_ref, average='macro'):.3f}"
          f" | local pipeline: {f1_score(test['label'], pred_loc, average='macro'):.3f}")
    if not n:
        print("  (notebook 06 reported test macro F1 0.500 for the XGBoost hybrid)")


if __name__ == "__main__":
    main()
