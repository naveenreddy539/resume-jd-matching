"""Streamlit app: rank resumes against a job description (runs locally).

Start from the project root:
    streamlit run app/streamlit_app.py
It opens in the browser at http://localhost:8501
"""
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # so "src" can be imported when run with streamlit
    sys.path.insert(0, str(ROOT))

from src.predict import Matcher  # noqa: E402

SAMPLE_DIR = ROOT / "app" / "samples"
MAX_RESUMES = 50
MIN_CHARS = 20
SHORT_WORDS = 150      # training resumes average ~640 words; much shorter texts are less reliable

st.set_page_config(page_title="Resume-JD Matcher", layout="wide")


@st.cache_resource(show_spinner="Loading the model (first run only)...")
def get_matcher():
    return Matcher()


def read_upload(f):
    if f.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(f)
        return "\n".join((p.extract_text() or "") for p in reader.pages).strip()
    return f.getvalue().decode("utf-8", errors="ignore").strip()


def load_sample():
    st.session_state["jd"] = (SAMPLE_DIR / "job_description.txt").read_text(encoding="utf-8")
    st.session_state["sample_resumes"] = {p.name: p.read_text(encoding="utf-8")
                                          for p in sorted(SAMPLE_DIR.glob("resume_*.txt"))}
    st.session_state.pop("output", None)


def clear_all():
    st.session_state["jd"] = ""
    st.session_state["pasted"] = ""
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.session_state.pop("sample_resumes", None)
    st.session_state.pop("output", None)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About the model")
    st.markdown(
        "**Model:** XGBoost hybrid using 14 engineered features (skill overlap, experience, "
        "TF-IDF and embedding similarity) plus 32 Sentence-BERT comparison components.\n\n"
        "**Test results** (1,759 pairs, 71 jobs not seen in training)\n"
        "- Macro F1 0.500, accuracy 0.548\n"
        "- Shortlisting: Precision@5 0.662, NDCG@5 0.781\n\n"
        "**Fit score** = 50 × (P(Potential Fit) + 2 × P(Good Fit)), from 0 to 100. "
        "Candidates are ranked by this score.\n\n"
        "**Limitations**\n"
        "- Good Fit is often under-predicted (recall 0.36)\n"
        "- Experience is estimated from job date ranges only\n"
        "- Short resumes are less reliable (training resumes average about 640 words)\n"
        "- Trained on one public dataset whose labels could not be verified"
    )

# ---------------------------------------------------------------- inputs
st.title("Resume-JD Matcher")
st.caption("Ranks resumes against a job description and shows matched and missing skills.")
st.info("Decision support only: use the ranking to prioritise reading, not to reject candidates "
        "automatically. The final decision stays with the recruiter.")

left, right = st.columns(2)
with left:
    st.subheader("1. Job description")
    st.text_area("Paste the job description", key="jd", height=320)
with right:
    st.subheader("2. Resumes")
    uploads = st.file_uploader("Upload resumes (PDF or TXT, several at once)", type=["pdf", "txt"],
                               accept_multiple_files=True,
                               key=f"uploader_{st.session_state.get('uploader_key', 0)}")
    st.text_area("Or paste resumes (separate each resume with a line containing only ---)",
                 key="pasted", height=140)
    if st.session_state.get("sample_resumes"):
        st.caption(f"Sample loaded: {len(st.session_state['sample_resumes'])} resumes from the test set "
                   "(true labels in app/samples/expected_labels.csv)")

b1, b2, b3 = st.columns([1, 1, 4])
b1.button("Load sample", on_click=load_sample)
b2.button("Clear", on_click=clear_all)
run = b3.button("Match resumes", type="primary")

# ---------------------------------------------------------------- run
if run:
    errors, warnings, resumes = [], [], {}
    resumes.update(st.session_state.get("sample_resumes", {}))
    for f in uploads or []:
        try:
            text = read_upload(f)
        except Exception as e:                       # corrupt or protected file
            warnings.append(f"Could not read {f.name} ({e}); skipped.")
            continue
        if len(text) < MIN_CHARS:
            warnings.append(f"{f.name} has no readable text (scanned PDF?); skipped.")
        elif f.name in resumes:
            errors.append(f"Duplicate file name: {f.name}")
        else:
            resumes[f.name] = text
    chunks = [c.strip() for c in re.split(r"(?m)^\s*---\s*$", st.session_state.get("pasted", ""))]
    for i, c in enumerate([c for c in chunks if c], 1):
        if len(c) < MIN_CHARS:
            warnings.append(f"Pasted resume {i} is too short; skipped.")
        else:
            resumes[f"Pasted resume {i}"] = c

    jd = st.session_state.get("jd", "").strip()
    if len(jd) < MIN_CHARS:
        errors.append("Please enter a job description (at least 20 characters).")
    if not resumes:
        errors.append("Please add at least one resume.")
    if len(resumes) > MAX_RESUMES:
        errors.append(f"At most {MAX_RESUMES} resumes at a time (got {len(resumes)}).")

    for w in warnings:
        st.warning(w)
    if errors:
        for e in errors:
            st.error(e)
        st.session_state.pop("output", None)
    else:
        with st.spinner(f"Scoring {len(resumes)} resume(s)..."):
            out = get_matcher().match(jd, resumes)
        out["words"] = {n: len(t.split()) for n, t in resumes.items()}
        st.session_state["output"] = out

# ---------------------------------------------------------------- results
out = st.session_state.get("output")
if out:
    res = out["results"]
    st.divider()
    st.subheader("3. Ranked candidates")

    m = st.columns(5)
    m[0].metric("Resumes scored", out["resumes_scored"])
    m[1].metric("Good Fit", sum(r["label"] == "Good Fit" for r in res))
    m[2].metric("Potential Fit", sum(r["label"] == "Potential Fit" for r in res))
    m[3].metric("No Fit", sum(r["label"] == "No Fit" for r in res))
    m[4].metric("Time", f"{out['elapsed_ms']:.0f} ms")

    table = pd.DataFrame([{
        "Rank": r["rank"], "Candidate": r["name"], "Prediction": r["label"],
        "Confidence (%)": round(100 * r["confidence"]), "Fit score": r["fit_score"],
        "Skill coverage (%)": round(100 * r["skill_coverage"]),
        "Semantic similarity": r["semantic_similarity"], "Missing skills": len(r["missing_skills"]),
    } for r in res])
    st.dataframe(table, hide_index=True, column_config={
        "Fit score": st.column_config.ProgressColumn("Fit score", min_value=0, max_value=100, format="%.1f"),
    })

    st.markdown("**Candidate details**")
    for r in res:
        with st.expander(f"#{r['rank']}  {r['name']}  —  {r['label']}  (fit score {r['fit_score']})",
                         expanded=r["rank"] == 1):
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"**Matched skills ({len(r['matched_skills'])}):** "
                           + (", ".join(r["matched_skills"]) or "none"))
                st.warning(f"**Missing skills ({len(r['missing_skills'])}):** "
                           + (", ".join(r["missing_skills"]) or "none"))
                req = r["jd_years_required"]
                have = r["resume_years"]
                st.write(f"**Experience:** required {'not stated' if req is None else f'{req:g} yrs'} | "
                         f"resume {'not found' if have is None else f'{have:g} yrs'} "
                         "(estimated from job date ranges)")
                st.write(f"**Semantic similarity:** {r['semantic_similarity']:.2f} | "
                         f"**Skill coverage:** {r['skill_coverage']:.0%}")
            with c2:
                st.write("**Class probabilities**")
                for label, p in r["probabilities"].items():
                    st.progress(float(p), text=f"{label}: {p:.0%}")
                if out["words"].get(r["name"], 0) < SHORT_WORDS:
                    st.caption(f"Short resume ({out['words'][r['name']]} words): prediction is less reliable.")

    csv = pd.DataFrame([{
        "rank": r["rank"], "candidate": r["name"], "prediction": r["label"], "confidence": r["confidence"],
        "p_no_fit": r["probabilities"]["No Fit"], "p_potential_fit": r["probabilities"]["Potential Fit"],
        "p_good_fit": r["probabilities"]["Good Fit"], "fit_score": r["fit_score"],
        "skill_coverage": r["skill_coverage"], "semantic_similarity": r["semantic_similarity"],
        "jd_years_required": r["jd_years_required"], "resume_years": r["resume_years"],
        "matched_skills": "; ".join(r["matched_skills"]), "missing_skills": "; ".join(r["missing_skills"]),
    } for r in res]).to_csv(index=False).encode("utf-8")
    st.download_button("Download results (CSV)", csv, "match_results.csv", "text/csv")
