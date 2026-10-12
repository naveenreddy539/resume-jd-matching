"""FastAPI endpoint for Resume-JD matching (runs locally).

Start from the project root:
    uvicorn app.api:app --port 8000
Then open http://127.0.0.1:8000/docs to try it in the browser (Swagger UI).

POST /match   one job description + one or more resumes -> ranked fit results
GET  /health  confirms the model is loaded
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import Matcher

MAX_RESUMES = 50
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["matcher"] = Matcher()          # load the model once at startup, not per request
    yield
    state.clear()


app = FastAPI(title="Resume-JD Matching API",
              description="Decision support for recruiters: predicts resume fit (No Fit / Potential Fit / "
                          "Good Fit), ranks candidates and shows matched and missing skills. "
                          "Not intended for automatic hiring decisions.",
              version="1.0", lifespan=lifespan)


class Resume(BaseModel):
    name: str = Field(..., min_length=1, examples=["Candidate A"])
    text: str = Field(..., min_length=20, examples=["Software engineer. Skills: Python, SQL, AWS, Docker. 01/2018 - Current."])


class MatchRequest(BaseModel):
    job_description: str = Field(..., min_length=20,
                                 examples=["We need a Python developer with 3+ years of experience in SQL, AWS and Docker."])
    resumes: list[Resume] = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "matcher" in state}


@app.post("/match")
def match(req: MatchRequest):
    if len(req.resumes) > MAX_RESUMES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_RESUMES} resumes per request")
    names = [r.name for r in req.resumes]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=400, detail="Resume names must be unique")
    return state["matcher"].match(req.job_description, {r.name: r.text for r in req.resumes})
