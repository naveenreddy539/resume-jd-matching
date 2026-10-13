# Resume Screening and Job Description Matching

An ML system that predicts how well a resume fits a job description (**No Fit / Potential Fit / Good Fit**), ranks candidates for a job, and shows **matched and missing skills** so every score can be explained. It runs locally as a Streamlit app and a FastAPI service.

> **Decision support only.** The ranking helps a recruiter decide which resumes to read first. It is not meant to reject candidates automatically; the final decision stays with a person.

Capstone project, Data Science & AI/ML programme (Batch 3), problem statement #5.

---

## Results at a glance

Final model: **XGBoost hybrid** (14 engineered features + 32 Sentence-BERT comparison components).
All numbers below are on the **untouched test set** (1,759 pairs, 71 job descriptions never seen in training).

| Metric | XGBoost hybrid | Keyword baseline (TF-IDF) | Majority class |
|---|---|---|---|
| Macro F1 | **0.500** | 0.368 | 0.218 |
| Accuracy | **0.548** | 0.423 | 0.487 |
| Precision@5 (shortlist) | **0.662** | 0.528 | – |
| NDCG@5 (shortlist) | **0.781** | 0.634 | – |

- About **2 of every 3 top-5 candidates are real fits** (Potential or Good Fit), against about half by chance (random ranking: Precision@5 0.486).
- Laptop CPU, no GPU: about **1 second for 4 full-length resumes** end to end; the model step alone takes 0.17 ms per pair.

---

## Dataset

[Resume-JD Match](https://huggingface.co/datasets/facehuggerapoorv/resume-jd-match) (Hugging Face): 6,241 training and 1,759 test records. Each record is one text field holding a job description and a resume, plus a 3-class label.

Key findings during data preparation:

- **Label leakage:** every training record ended with the answer (`"The result is, No Fit"`); test records did not. The embedded label was removed. Training on the raw text would have produced near-perfect training scores and failed on test.
- **Label noise:** 6 JD–resume pairs (12 rows) appeared with conflicting labels and were removed, plus 1 exact duplicate.
- **Structure:** only 351 unique JDs and about 640 unique resumes. Test JDs never appear in training, while 99.9% of test resumes do, so the task is judging resumes for **unseen jobs**.
- **Length:** the average record is about 8,600 characters; 98.4% exceed BERT's 512-token limit.

Splits used for all models (validation holds out whole JDs, like the test set):

| Split | Rows | Job descriptions |
|---|---|---|
| Train | 5,285 | 240 |
| Validation | 943 | 40 |
| Test | 1,759 | 71 |

---

## Approach

1. **Cleaning:** normalise special characters; remove emails, URLs and phone numbers; split words stuck together by lost line breaks (`SummaryHighly` → `Summary Highly`) while protecting real terms like `JavaScript` and `PostgreSQL`.
2. **Features** (14 per JD–resume pair):
   - **Skills:** a 203-skill list mined from training data only; skill coverage (share of the JD's skills found in the resume), Jaccard overlap, matched / missing counts.
   - **Experience:** years required by the JD vs years estimated from the resume's job date ranges, gap, meets/doesn't meet, missing flags.
   - **Text similarity:** TF-IDF cosine (keywords) and Sentence-BERT cosine (meaning; `all-MiniLM-L6-v2`, long texts split into 180-word chunks and averaged).
3. **Embedding comparisons:** for each pair, |JD − resume| and JD × resume (768 values), compressed to 32 with PCA. The raw resume vector is never used, so the model cannot memorise individual resumes.
4. **Models**, from simple to advanced, compared on validation (macro F1):

| Model | Validation macro F1 | Test macro F1 |
|---|---|---|
| Majority class | 0.218 | 0.218 |
| Keyword baseline (TF-IDF cosine) | 0.408 | 0.368 |
| Resume-only check (no JD) | 0.426 | 0.441 |
| Logistic Regression (14 features) | 0.463 | 0.431 |
| Random Forest (14 features) | 0.498 | 0.458 |
| Sentence-BERT model (embedding comparisons) | 0.527 | 0.443 |
| **XGBoost hybrid (14 features + 32 PCA)** | **0.545** | **0.500** |
| DistilBERT cross-encoder (fine-tuned, 512 tokens) | 0.373 | 0.352 |

- **Ablation:** XGBoost on the 14 features alone scored 0.497 on validation; adding the embedding components raised it to 0.545.
- **DistilBERT** underperformed: it reads only 512 tokens of pairs averaging about 1,000 words, and 240 training JDs are too few to learn JD–resume relationships from scratch.
- **Resume-only check:** a model that never sees the JD still reaches 0.441 on test, because resumes are reused across many jobs. The hybrid beats it by 0.059 in classification and by 0.13 NDCG@5 in ranking, where comparing the JD with the resume matters most.

All training runs are logged in `reports/experiments.csv`.

---

## Demo: 3 sample predictions and 1 failure case

The app's **Load sample** button loads one real test job description (an accounting role) and four of its resumes (`app/samples/`, true labels in `expected_labels.csv`):

| Rank | Resume | Prediction | Actual | Fit score | Case |
|---|---|---|---|---|---|
| 1 | resume_2 | Good Fit | Good Fit | 77.1 | correct |
| 2 | resume_3 | Potential Fit | Potential Fit | 61.9 | correct |
| 3 | resume_4 | No Fit | Good Fit | 41.1 | **failure case** |
| 4 | resume_1 | No Fit | No Fit | 4.1 | correct |

**Failure case (resume_4):** labelled Good Fit, predicted No Fit. It matches only 2 skills (accounting, excel), misses the JD's core duties (audit, compliance, balance sheet, financial statements), has an estimated 4 years against 5 required, and has lower semantic similarity (0.41) than the other candidates. The model was also unsure (No Fit 54%, Good Fit 36%). Its fit score still ranked it **above** the true No Fit resume, so a recruiter reading the top 3 would still see it, which is why the tool is for shortlisting rather than automatic rejection.

---

## Setup

Tested with Python 3.14 on Windows.

```bash
git clone https://github.com/naveenreddy539/resume-jd-matching.git
cd resume-jd-matching
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB) to the Hugging Face cache; later runs work offline.

## Run the app (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501. Paste a job description, upload resumes (PDF or TXT) or paste them, and click **Match resumes**. Click **Load sample** for the demo set above. Results can be downloaded as CSV.

## Run the API (FastAPI)

```bash
uvicorn app.api:app --port 8000
```

Interactive docs at http://127.0.0.1:8000/docs.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Confirms the model is loaded |
| POST | `/match` | One job description + 1–50 resumes → ranked results |

Example request body for `POST /match`:

```json
{"job_description": "We need a Python developer with 3+ years of experience in SQL, AWS and Docker.", "resumes": [{"name": "Candidate A", "text": "Software engineer. Skills: Python, SQL, Docker, AWS, Git. Developer, 01/2018 - Current."}, {"name": "Candidate B", "text": "Sales Associate with customer service and retail experience. Store Associate, 2015 - 2019."}]}
```

Each result contains the predicted label, confidence, class probabilities, a 0–100 fit score, skill coverage, semantic similarity, experience figures, and matched and missing skills, ranked best first.

**Fit score** = 50 × (P(Potential Fit) + 2 × P(Good Fit)), the same expected-fit score used to rank candidates in the test evaluation.

## Rebuild the data splits

```bash
python -m src.data_loader
```

Downloads the dataset and recreates the cleaned train / validation / test splits in `data/`. The validation split uses the exact job descriptions from the notebooks (`data/val_jd_hashes.json`), because the same random seed gave a different split with a newer scikit-learn version. Output: 5,285 / 943 / 1,759 rows.

## Consistency check (optional)

```bash
python -m src.verify
```

Recomputes every feature for all 1,759 test pairs on the local machine and compares them with the values from training (needs `data/test_feat.parquet` and `data/embeddings.joblib` exported from the notebooks; not included in the repository). Result on the development laptop: embeddings cosine 1.000000, all 14 features identical, 100% identical predictions, test macro F1 0.500. This rules out training–serving skew.

---

## Project structure

```
resume-jd-matching/
├── notebooks/               # run in Google Colab, in order (data saved to Google Drive)
│   ├── 01_data_exploration.ipynb      # load, inspect, split JD/resume, remove embedded label
│   ├── 02_eda_preprocessing.ipynb     # cleaning, duplicates, validation split, EDA charts
│   ├── 03_feature_engineering.ipynb   # skill list, skill / experience / similarity features
│   ├── 04_baseline_models.ipynb       # majority, keyword, resume-only, LogReg, Random Forest
│   ├── 05_advanced_models.ipynb       # Sentence-BERT model, XGBoost hybrid, DistilBERT
│   └── 06_evaluation.ipynb            # test set: metrics, ranking, latency, errors, fairness
├── src/
│   ├── preprocess.py        # cleaning, skill and experience extraction (copied from the notebooks)
│   ├── predict.py           # Matcher: loads the model once, scores and ranks resumes
│   ├── data_loader.py       # rebuilds the data splits from Hugging Face
│   ├── verify.py            # consistency check against the training features
│   └── make_samples.py      # builds the demo set in app/samples/
├── app/
│   ├── streamlit_app.py     # recruiter-facing app
│   ├── api.py               # FastAPI service
│   └── samples/             # demo job description, 4 resumes, expected labels
├── models/                  # final model, version-independent formats
│   ├── xgb_hybrid.json      # XGBoost model
│   ├── xgb_hybrid_meta.json # feature and class order
│   ├── pca_32.npz           # PCA mean and components
│   ├── tfidf.json           # TF-IDF vocabulary and idf weights (fitted on training texts only)
│   └── skills.csv           # 203-skill list with variants
├── data/val_jd_hashes.json  # exact validation job descriptions
├── reports/                 # experiment log, test results, figures
├── .streamlit/config.toml
└── requirements.txt
```

Model files are stored as plain data (JSON / NumPy) rather than pickles, because training used scikit-learn 1.6.1 and the app runs on 1.9.1; pickled objects can break across versions.

---

## Fairness and responsible use

- **No protected attributes** are used. Features are only comparisons between a JD and a resume (skills, experience, text similarity). Resumes in the dataset appear anonymised (placeholder emails and phone numbers).
- **Experience can act as a proxy for age**, so performance was compared across experience bands on the test set: macro F1 0.506 (0–5 years), 0.491 (6–10), 0.506 (10+). Good Fit is slightly under-predicted for 0–10 years (about 21% predicted vs 27–28% actual) and slightly over-predicted for 10+ years (20.5% vs 17.3%).
- Resumes whose experience cannot be estimated perform worse (macro F1 0.328, 51 rows) and should go to human review.
- The app shows a decision-support notice and explains each score with matched and missing skills.

## Limitations

- **Good Fit is often under-predicted** (test recall 0.36); most errors are between neighbouring classes.
- **Labels are partly tied to the resume itself** (the resume-only check reaches 0.441), and some labels look noisy; there is no dataset card describing how labels were assigned.
- **Experience** is estimated only from job date ranges, and "Current" is taken as the latest year in that resume, so a stated total such as "6 years of experience" is not read. The app keeps this rule because the model was trained with it.
- **Skill extraction** is keyword-based: the skill "legal" can be picked up from JD boilerplate about work authorisation.
- **Short resumes** are less reliable; training resumes average about 640 words (the app flags texts under 150 words).
- PCA keeps 30% of the variance of the embedding comparisons; more components is possible future work.
- Trained on one public dataset; results may not transfer to other industries or resume formats.

## Future work

- Read stated experience totals and use the dataset's collection year for "Current".
- Context-aware skill extraction (ignore boilerplate sections).
- A cross-encoder reranker on long-text models, applied to the top candidates only.
- Calibrated probabilities and a tuned decision threshold for Good Fit.
