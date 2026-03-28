from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Dict, Any
import pandas as pd
import joblib

# Load model once at startup
try:
    model = joblib.load("best_model.pkl")
except Exception as exc:
    raise RuntimeError("Could not load best_model.pkl. Ensure it exists in project root.") from exc

app = FastAPI(
    title="Student Exam Score Predictor API",
    version="1.0",
    description="ML API endpoint for the Student Exam Score Predictor app."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentData(BaseModel):
    age: int
    gender: Literal["Male", "Female", "Other"]
    study_hours_per_day: float
    social_media_hours: float
    netflix_hours: float
    attendance_percentage: float
    sleep_hours: float
    exercise_frequency: int
    mental_health_rating: int
    part_time_job: Literal["No", "Yes"]
    diet_quality: Literal["Poor", "Average", "Good"]
    parental_education_level: Literal["High School", "Bachelor", "Master"]
    internet_quality: Literal["Poor", "Average", "Good"]
    extracurricular_participation: Literal["No", "Yes"]


def _build_input_df(payload: StudentData) -> pd.DataFrame:
    df = pd.DataFrame([payload.dict()])
    df = pd.get_dummies(
        df,
        columns=[
            "gender",
            "part_time_job",
            "diet_quality",
            "parental_education_level",
            "internet_quality",
            "extracurricular_participation",
        ],
        drop_first=True,
    )

    expected_features = [
        "age", "study_hours_per_day", "social_media_hours", "netflix_hours",
        "attendance_percentage", "sleep_hours", "exercise_frequency", "mental_health_rating",
        "gender_Male", "gender_Other",
        "part_time_job_Yes",
        "diet_quality_Good", "diet_quality_Poor",
        "parental_education_level_High School", "parental_education_level_Master",
        "internet_quality_Good", "internet_quality_Poor",
        "extracurricular_participation_Yes",
    ]

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]
    return df


def _gpa_letter(pred_score: float) -> Dict[str, Any]:
    if pred_score >= 90:
        return {"gpa": 4.0, "grade": "A+"}
    if pred_score >= 80:
        return {"gpa": 3.6, "grade": "A"}
    if pred_score >= 70:
        return {"gpa": 3.2, "grade": "B+"}
    if pred_score >= 60:
        return {"gpa": 2.8, "grade": "B"}
    if pred_score >= 50:
        return {"gpa": 2.4, "grade": "C+"}
    if pred_score >= 40:
        return {"gpa": 2.0, "grade": "C"}
    if pred_score >= 30:
        return {"gpa": 1.8, "grade": "D+"}
    if pred_score >= 20:
        return {"gpa": 1.6, "grade": "D"}
    return {"gpa": 0.8, "grade": "F"}


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "Student Exam Score Predictor API",
        "predict_endpoint": "/api/predict",
        "description": "POST student payload to /api/predict for score prediction (JSON)."
    }


@app.post("/api/predict")
def predict(data: StudentData) -> Dict[str, Any]:
    try:
        input_df = _build_input_df(data)
        pred_score = float(model.predict(input_df)[0])
        pred_score = max(0.0, min(100.0, pred_score))

        gpa_info = _gpa_letter(pred_score)

        result = {
            "pred_score": round(pred_score, 2),
            "gpa": gpa_info["gpa"],
            "letter_grade": gpa_info["grade"],
            "interpretation": (
                "Excellent" if pred_score >= 90 else
                "Very Good" if pred_score >= 80 else
                "Good" if pred_score >= 70 else
                "Average" if pred_score >= 60 else
                "Below Average"
            ),
            "confidence": min(99, max(50, pred_score - 40)),
            "percentile": round(pred_score, 1)
        }
        return result

    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
