"""
Prediction helper for the saved salary model package.
"""

from __future__ import annotations

from typing import Dict

import joblib
import pandas as pd

from preprocessing import add_engineered_features


MODEL_PATH = "models/salary_model.pkl"


def load_model(model_path: str = MODEL_PATH) -> Dict:
    """Load saved model package."""
    return joblib.load(model_path)


def predict_salary(input_data: Dict, model_path: str = MODEL_PATH) -> float:
    """
    Predict salary for a single candidate/job profile.

    input_data example:
    {
        "job_title": "Data Scientist",
        "experience_years": 5,
        "education_level": "Master",
        "skills_count": 10,
        "industry": "Technology",
        "company_size": "Large",
        "location": "USA",
        "remote_work": "Hybrid",
        "certifications": 2
    }
    """
    model_package = load_model(model_path)
    selected_features = model_package["selected_features"]
    pipeline = model_package["pipeline"]

    input_df = pd.DataFrame([input_data])
    input_df = add_engineered_features(input_df)
    input_df = input_df[selected_features]

    prediction = pipeline.predict(input_df)[0]
    return float(prediction)


if __name__ == "__main__":
    sample = {
        "job_title": "Data Scientist",
        "experience_years": 5,
        "education_level": "Master",
        "skills_count": 10,
        "industry": "Technology",
        "company_size": "Large",
        "location": "USA",
        "remote_work": "Hybrid",
        "certifications": 2,
    }

    salary = predict_salary(sample)
    print(f"Predicted salary: ${salary:,.2f}")
