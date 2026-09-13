"""
Streamlit MVP app for salary prediction.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import add_engineered_features, map_experience_level


MODEL_PATH = "models/salary_model.pkl"


@st.cache_resource
def load_model_package():
    """Load the saved model package once."""
    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Job Salary Prediction MVP",
    page_icon="💼",
    layout="centered",
)

st.title("Job Salary Prediction")
# st.write(
#     "Academic MVP for predicting salary from structured job/resume profile data. "
#     "The project uses a Genetic Algorithm as a biologically inspired feature-selection method."
# )

try:
    model_package = load_model_package()
except FileNotFoundError:
    st.error(
        "Model file not found. Please run `python src/train_model.py` first "
        "from the project root directory."
    )
    st.stop()

selected_features = model_package["selected_features"]
pipeline = model_package["pipeline"]
metadata = model_package.get("metadata", {})

st.subheader("Candidate / job profile input")

job_title = st.selectbox(
    "Job title",
    [
        "AI Engineer",
        "Data Analyst",
        "Frontend Developer",
        "Business Analyst",
        "Product Manager",
        "Backend Developer",
        "Machine Learning Engineer",
        "DevOps Engineer",
        "Software Engineer",
        "Cybersecurity Analyst",
        "Data Scientist",
        "Cloud Engineer",
    ],
)

experience_years = st.slider("Experience years", 0, 20, 5)
experience_level_group = map_experience_level(experience_years)
st.caption(f"Derived experience level: **{experience_level_group}**")

education_level = st.selectbox(
    "Education level",
    ["High School", "Diploma", "Bachelor", "Master", "PhD"],
)

skills_count = st.slider("Skills count", 0, 19, 8)

industry = st.selectbox(
    "Industry",
    [
        "Healthcare",
        "Telecom",
        "Media",
        "Retail",
        "Manufacturing",
        "Education",
        "Finance",
        "Technology",
        "Consulting",
        "Government",
    ],
)

company_size = st.selectbox(
    "Company size",
    ["Startup", "Small", "Medium", "Large", "Enterprise"],
)

location = st.selectbox(
    "Location",
    [
        "India",
        "Australia",
        "Singapore",
        "Canada",
        "Sweden",
        "USA",
        "Netherlands",
        "Remote",
        "Germany",
        "UK",
    ],
)

remote_work = st.selectbox(
    "Remote work",
    ["Yes", "No", "Hybrid"],
)

certifications = st.slider("Certifications", 0, 5, 1)

input_data = {
    "job_title": job_title,
    "experience_years": experience_years,
    "education_level": education_level,
    "skills_count": skills_count,
    "industry": industry,
    "company_size": company_size,
    "location": location,
    "remote_work": remote_work,
    "certifications": certifications,
}

if st.button("Predict Salary"):
    input_df = pd.DataFrame([input_data])
    input_df = add_engineered_features(input_df)
    prediction = pipeline.predict(input_df[selected_features])[0]

    st.success(f"Predicted salary: ${prediction:,.2f}")

    if metadata.get("use_experience_calibrated_target"):
        st.caption(
            "This prediction uses the experience-calibrated target, because the original "
            "synthetic dataset assigns unrealistically high salaries to some junior profiles."
        )

st.subheader("Genetic Algorithm selected features")
st.write(selected_features)

st.subheader("Experience-level feature engineering")
st.write(
    "The app derives `experience_level_group` from `experience_years`: "
    "Junior = 0-2, Mid = 3-5, Senior = 6-10, Expert = 11+ years. "
    "This engineered feature can also be selected by the Genetic Algorithm."
)

st.subheader("Salary target calibration")
if metadata.get("use_experience_calibrated_target"):
    st.write(
        "The model was trained on an experience-calibrated salary target. "
        "This correction reduces unrealistic salaries for 0-9 years of experience "
        "and disappears for 10+ years of experience."
    )
else:
    st.write("The model was trained on the original salary column from the CSV dataset.")

st.subheader("Model explanation")
st.write(
    "The saved model is a scikit-learn pipeline. It keeps only the features selected "
    "by the Genetic Algorithm, preprocesses numeric and categorical variables, and "
    "then predicts salary using a tree-based regression model."
)

if "final_metrics" in metadata:
    st.subheader("Final model metrics")
    st.json(metadata["final_metrics"])

# st.warning(
#     "Disclaimer: this is an academic MVP for a university project. "
#     "It must not be used as a real HR compensation or hiring decision system."
# )
