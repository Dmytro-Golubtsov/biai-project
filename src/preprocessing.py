"""
Preprocessing utilities for the Job Salary Prediction MVP.

This module builds a scikit-learn preprocessing pipeline:
- feature engineering for experience levels based on years of experience
- StandardScaler for numeric features
- OneHotEncoder for categorical features
- ColumnTransformer to combine both transformations
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "salary"

RAW_FEATURES: List[str] = [
    "job_title",
    "experience_years",
    "education_level",
    "skills_count",
    "industry",
    "company_size",
    "location",
    "remote_work",
    "certifications",
]

ENGINEERED_FEATURES: List[str] = [
    "experience_level_group",
]

ALL_FEATURES: List[str] = RAW_FEATURES + ENGINEERED_FEATURES

NUMERIC_FEATURES: List[str] = [
    "experience_years",
    "skills_count",
    "certifications",
]

CATEGORICAL_FEATURES: List[str] = [
    "job_title",
    "education_level",
    "industry",
    "company_size",
    "location",
    "remote_work",
    "experience_level_group",
]


EXPERIENCE_LEVEL_ORDER: List[str] = ["Junior", "Mid", "Senior", "Expert"]


def map_experience_level(years: float | int) -> str:
    """
    Convert numeric years of experience into an interpretable experience level.

    Levels used in this project:
    - Junior: 0-2 years
    - Mid: 3-5 years
    - Senior: 6-10 years
    - Expert: 11+ years
    """
    years = float(years)

    if years <= 2:
        return "Junior"
    if years <= 5:
        return "Mid"
    if years <= 10:
        return "Senior"
    return "Expert"


def salary_experience_calibration_factor(years: float | int) -> float:
    """
    Return a realism correction factor for the synthetic salary target.

    The original dataset contains unrealistically high salaries for zero-experience
    candidates. For example, 0-year Software Engineers have an average salary above
    100k USD. This factor is used only during training to create a more realistic
    MVP target while keeping the original dataset unchanged.

    Factor logic:
    - 0 years  -> 0.55
    - 5 years  -> 0.775
    - 10+ years -> 1.00

    It is a simple linear correction that gradually disappears with experience.
    """
    years = max(0.0, float(years))
    if years >= 10:
        return 1.0
    return 0.55 + 0.45 * (years / 10.0)


def add_calibrated_salary_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    output_column: str = "salary_calibrated",
) -> pd.DataFrame:
    """
    Add an experience-calibrated salary target for a more realistic MVP demo.

    This does not overwrite the original `salary` column. The training script can
    decide whether to train on the original salary target or on `salary_calibrated`.
    """
    df_out = df.copy()

    if target_column not in df_out.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
    if "experience_years" not in df_out.columns:
        raise ValueError("Column 'experience_years' is required for salary calibration.")

    factors = df_out["experience_years"].apply(salary_experience_calibration_factor)
    df_out[output_column] = df_out[target_column] * factors
    return df_out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features used by both training and prediction.

    The original `experience_years` numeric feature is kept. The additional
    `experience_level_group` categorical feature gives the model a human-readable
    seniority signal derived from the same numeric experience column.
    """
    df_out = df.copy()

    if "experience_years" not in df_out.columns:
        raise ValueError("Column 'experience_years' is required to create engineered features.")

    df_out["experience_level_group"] = df_out["experience_years"].apply(map_experience_level)
    return df_out


def load_dataset(path: str = "data/job_salary_prediction_dataset.csv") -> pd.DataFrame:
    """Load the salary dataset from CSV and add engineered features."""
    df = pd.read_csv(path)
    return add_engineered_features(df)


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into X features and y target."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def get_feature_types(selected_features: Iterable[str] | None = None) -> Tuple[List[str], List[str]]:
    """
    Return numeric and categorical features for a selected feature subset.

    The Genetic Algorithm selects original and engineered feature groups, not
    one-hot encoded columns.
    """
    selected = list(selected_features) if selected_features is not None else ALL_FEATURES

    numeric = [col for col in NUMERIC_FEATURES if col in selected]
    categorical = [col for col in CATEGORICAL_FEATURES if col in selected]
    return numeric, categorical


def build_preprocessor(selected_features: Iterable[str] | None = None) -> ColumnTransformer:
    """Create ColumnTransformer for selected original/engineered features."""
    numeric, categorical = get_feature_types(selected_features)

    transformers = []

    if numeric:
        transformers.append(("num", StandardScaler(), numeric))

    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        )

    if not transformers:
        raise ValueError("At least one feature must be selected for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Create train/test split with fixed random state."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Return feature names after preprocessing."""
    names: List[str] = []

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder" or transformer == "drop":
            continue

        if transformer_name == "num":
            names.extend(list(columns))
        elif transformer_name == "cat":
            cat_transformer = preprocessor.named_transformers_["cat"]
            names.extend(cat_transformer.get_feature_names_out(columns).tolist())

    return names
