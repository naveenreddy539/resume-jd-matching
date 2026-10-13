"""Build the demo set for the app from the test data (run once).

Picks one test job description with:
  - 3 resumes the model classifies correctly (one No Fit, one Potential Fit, one Good Fit, if available)
  - 1 resume the model gets wrong (a clear miss preferred: Good Fit predicted as No Fit, or the reverse)
and writes them to app/samples/ as plain text, with the true labels in expected_labels.csv.

These are the "3 sample predictions + 1 failure case" required by the project brief.
The texts come from the public, anonymised Resume-JD Match test set.

Run from the project root:   python -m src.make_samples
"""
import joblib
import numpy as np
import pandas as pd

from src.predict import ROOT, Matcher

SAMPLE_DIR = ROOT / "app" / "samples"


def pick(test, order, min_labels):
    best = None
    for jd, g in test.groupby("jd"):
        ok, bad = g[g["pred"] == g["label"]], g[g["pred"] != g["label"]]
        if bad.empty or ok["label"].nunique() < min_labels:
            continue
        big = bad[((bad["label"] == "Good Fit") & (bad["pred"] == "No Fit")) |
                  ((bad["label"] == "No Fit") & (bad["pred"] == "Good Fit"))]
        miss = (big if not big.empty else bad).sort_values("conf", ascending=False).iloc[0]
        picks = [ok[ok["label"] == l].sort_values("conf", ascending=False).iloc[0]
                 for l in order if (ok["label"] == l).any()]
        if len(picks) < 3:   # fewer than 3 labels: fill up with the next most confident correct rows
            extra = ok[~ok.index.isin([p.name for p in picks])].sort_values("conf", ascending=False)
            picks += [extra.iloc[i] for i in range(min(3 - len(picks), len(extra)))]
        if len(picks) < 3 or miss["resume"] in {p["resume"] for p in picks}:
            continue
        score = (not big.empty, len({p["label"] for p in picks}), sum(p["conf"] for p in picks))
        if best is None or score > best[0]:
            best = (score, jd, picks, miss)
    return best


def main():
    test = pd.read_parquet(ROOT / "data" / "test_feat.parquet")
    E = joblib.load(ROOT / "data" / "embeddings.joblib")
    emb = dict(zip(E["texts"], E["embeddings"]))
    m = Matcher()

    # Predictions exactly as in the test evaluation (same model, same features)
    J = np.stack(test["jd"].map(emb).values)
    R = np.stack(test["resume"].map(emb).values)
    P = m.predict_proba(test[m.features], np.hstack([np.abs(J - R), J * R]))
    test["pred"], test["conf"] = np.array(m.order)[P.argmax(1)], P.max(1)
    test = test[test["resume"].str.split().str.len().between(200, 900)]   # realistic, readable lengths

    best = pick(test, m.order, 3) or pick(test, m.order, 2)
    if best is None:
        raise SystemExit("No suitable job description found")
    _, jd, picks, miss = best

    rows = [(p, "correct") for p in picks] + [(miss, "failure case")]
    rng = np.random.default_rng(7)
    rows = [rows[i] for i in rng.permutation(len(rows))]   # mixed order, so the ranking is visible in the demo

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLE_DIR / "job_description.txt").write_text(jd, encoding="utf-8")
    expected = []
    for i, (r, case) in enumerate(rows, 1):
        name = f"resume_{i}.txt"
        (SAMPLE_DIR / name).write_text(r["resume"], encoding="utf-8")
        expected.append({"file": name, "actual_label": r["label"],
                         "predicted_in_evaluation": r["pred"], "case": case})
    pd.DataFrame(expected).to_csv(SAMPLE_DIR / "expected_labels.csv", index=False)

    # Run the files through the same path the app uses (raw text -> Matcher.match)
    resumes = {e["file"]: (SAMPLE_DIR / e["file"]).read_text(encoding="utf-8") for e in expected}
    out = m.match((SAMPLE_DIR / "job_description.txt").read_text(encoding="utf-8"), resumes)
    exp = {e["file"]: e for e in expected}
    print(f"JD ({len(jd.split())} words): {jd[:150]}...\n")
    print(f"{'rank':<5}{'file':<15}{'app prediction':<16}{'actual':<15}{'fit score':<10}case")
    for r in out["results"]:
        e = exp[r["name"]]
        print(f"{r['rank']:<5}{r['name']:<15}{r['label']:<16}{e['actual_label']:<15}{r['fit_score']:<10}{e['case']}")
    same = all(r["label"] == exp[r["name"]]["predicted_in_evaluation"] for r in out["results"])
    print("\nApp predictions match the test evaluation:", same)
    print("Saved to", SAMPLE_DIR)


if __name__ == "__main__":
    main()
