"""Shared text processing for the Resume-JD Matching project.

Every function here is copied exactly from the notebooks, so the app computes
features the same way the model was trained (no training-serving skew):
  - Notebook 01: JD / resume split pattern
  - Notebook 02: clean_text (cleaning rules 1-5)
  - Notebook 03: skill extraction and experience extraction
"""
import re

import numpy as np

# ---------------------------------------------------------------------------
# Notebook 01: split each record into JD and resume, and drop the embedded label
# Format: "For the given job description <<JD>> the resume: <<RESUME>>. The result is, LABEL"
# ---------------------------------------------------------------------------
SPLIT_PATTERN = r"^For the given job description <<(.*)>> the resume: <<(.*)>>\. The result is,\s*(.*?)\s*$"

# ---------------------------------------------------------------------------
# Notebook 02: text cleaning
# ---------------------------------------------------------------------------
PROTECTED = ["JavaScript", "MySQL", "PowerPoint", "SharePoint", "PostgreSQL", "jQuery", "DevOps", "iOS",
             "QuickBooks", "MongoDB", "NoSQL", "MapReduce", "AutoCAD", "IoT", "WebDriver", "PySpark",
             "TestNG", "AngularJS", "GitHub", "SaaS", "DynamoDB", "TypeScript", "BigQuery", "MatLab",
             "ProvideX", "PeopleSoft", "CloudWatch", "IntelliJ", "WinForms", "CloudFormation"]


def camel_split(s):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", s)


# After splitting, join protected terms back together (e.g. "Java Script" -> "JavaScript")
RESTORE = [(re.compile(r"\b" + re.escape(camel_split(t)) + r"\b"), t) for t in PROTECTED]


def clean_text(s):
    s = s.replace("\xa0", " ").replace("–", "-").replace("—", "-")       # rule 1
    s = re.sub(r"\[[^\]]*\]\(mailto:[^)]*\)", " ", s)                    # rule 2: markdown mailto links
    s = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", " ", s)                       # rule 2: emails
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)                         # rule 2: URLs
    s = re.sub(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", " ", s)           # rule 2: phones
    s = camel_split(s)                                                   # rule 3: split stuck words
    for pat, term in RESTORE:                                            # rule 3: keep protected terms
        s = pat.sub(term, s)
    return re.sub(r"\s+", " ", s).strip()                                # rule 4 (case kept: rule 5)


# ---------------------------------------------------------------------------
# Notebook 03: skill extraction (203-skill list in models/skills.csv)
# ---------------------------------------------------------------------------
def build_skill_patterns(skills_df):
    """One whole-term pattern per skill, matching any of its variants (works for c++, c#, .net)."""
    return {
        row.skill: re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(v) for v in row.variants.split("|")) + r")(?![a-z0-9])")
        for row in skills_df.itertuples()
    }


def extract_skills(text, patterns):
    t = text.lower()
    return {s for s, p in patterns.items() if p.search(t)}


# ---------------------------------------------------------------------------
# Notebook 03: experience extraction
# ---------------------------------------------------------------------------
# JD required years (rules 1-3)
YRS = r"(\d{1,2})\s*\+?\s*(?:(?:-|to)\s*\d{1,2}\s*\+?\s*)?(?:years?|yrs?)\b"
MIN_YRS = re.compile(r"(?:minimum(?: of)?|at least|min\.?)\s*" + YRS)
ANY_YRS = re.compile(YRS)


def jd_years(text):
    t = text.lower()
    found = [int(m.group(1)) for m in MIN_YRS.finditer(t)]
    for m in ANY_YRS.finditer(t):
        window = t[max(0, m.start() - 40):m.end() + 40]
        if "experience" in window:
            found.append(int(m.group(1)))          # group 1 = lower number of a range
    found = [y for y in found if 0 <= y <= 30]
    return max(found) if found else np.nan        # rule 3: highest requirement


# Resume years from job date ranges (rules 4-6)
MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
DATE = rf"(?<!\d)(?:(?:0?[1-9]|1[0-2])/|{MONTHS}\s+)?((?:19|20)\d{{2}})(?!\d)"
RANGE = re.compile(rf"{DATE}\s*(?:-|to)\s*(?:{DATE}|(current|present|now))")


def resume_years(text):
    t = text.lower()
    ranges = []
    for m in RANGE.finditer(t):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else None      # None = Current/Present
        if not 1960 <= start <= 2025:                      # rule 6
            continue
        if end is not None and (not 1960 <= end <= 2025 or end < start):
            continue
        ranges.append((start, end))
    if not ranges:
        return np.nan
    latest = max([s for s, _ in ranges] + [e for _, e in ranges if e])   # rule 5
    return float(max(e if e else latest for _, e in ranges) - min(s for s, _ in ranges))
