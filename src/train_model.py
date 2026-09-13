"""
Train baseline salary prediction models, run Genetic Algorithm feature selection,
train final model, evaluate it, explain it, and save the final pipeline.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from genetic_feature_selection import GeneticFeatureSelector
from preprocessing import (
    ALL_FEATURES,
    TARGET_COLUMN,
    add_calibrated_salary_target,
    build_preprocessor,
    get_transformed_feature_names,
    load_dataset,
    make_train_test_split,
    split_features_target,
)


DATA_PATH = "data/job_salary_prediction_dataset.csv"
MODEL_PATH = "models/salary_model.pkl"
RESULTS_PATH = "models/training_results.json"

# The original synthetic dataset has unrealistically high salaries for candidates
# with 0-2 years of experience. For a more believable MVP demo, train on an
# experience-calibrated target. Set this to False if you want to train strictly
# on the original CSV salary column.
USE_EXPERIENCE_CALIBRATED_TARGET = True
CALIBRATED_TARGET_COLUMN = "salary_calibrated"


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate MAE, RMSE and R²."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def train_and_evaluate_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_features=None,
) -> Tuple[Pipeline, Dict[str, float]]:
    """Train a model pipeline and evaluate it."""
    features = list(selected_features) if selected_features is not None else ALL_FEATURES

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("model", model),
        ]
    )

    pipeline.fit(X_train[features], y_train)
    predictions = pipeline.predict(X_test[features])
    metrics = regression_metrics(y_test, predictions)

    print(f"\n{model_name}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    return pipeline, metrics


def get_feature_importance(final_pipeline: Pipeline, selected_features) -> pd.DataFrame:
    """Return transformed feature importances for tree-based models."""
    model = final_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    preprocessor = final_pipeline.named_steps["preprocessor"]
    transformed_names = get_transformed_feature_names(preprocessor)

    importance_df = pd.DataFrame(
        {
            "feature": transformed_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance_df


def aggregate_importance(importance_df: pd.DataFrame, selected_features) -> pd.DataFrame:
    """Aggregate one-hot encoded feature importances back to original features."""
    rows = []

    for original_feature in selected_features:
        mask = (
            (importance_df["feature"] == original_feature)
            | (importance_df["feature"].str.startswith(f"{original_feature}_"))
        )
        rows.append(
            {
                "feature_group": original_feature,
                "importance": float(importance_df.loc[mask, "importance"].sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def main() -> None:
    os.makedirs("models", exist_ok=True)

    df = load_dataset(DATA_PATH)

    original_zero_exp_mean = df.loc[df["experience_years"] == 0, TARGET_COLUMN].mean()
    original_zero_exp_swe_mean = df.loc[
        (df["experience_years"] == 0) & (df["job_title"] == "Software Engineer"),
        TARGET_COLUMN,
    ].mean()

    if USE_EXPERIENCE_CALIBRATED_TARGET:
        df = add_calibrated_salary_target(df, target_column=TARGET_COLUMN, output_column=CALIBRATED_TARGET_COLUMN)
        training_target = CALIBRATED_TARGET_COLUMN

        calibrated_zero_exp_mean = df.loc[df["experience_years"] == 0, training_target].mean()
        calibrated_zero_exp_swe_mean = df.loc[
            (df["experience_years"] == 0) & (df["job_title"] == "Software Engineer"),
            training_target,
        ].mean()

        print("Using experience-calibrated salary target for training.")
        print(f"Original mean salary for 0 years experience: ${original_zero_exp_mean:,.2f}")
        print(f"Calibrated mean salary for 0 years experience: ${calibrated_zero_exp_mean:,.2f}")
        print(f"Original mean salary for 0-year Software Engineer: ${original_zero_exp_swe_mean:,.2f}")
        print(f"Calibrated mean salary for 0-year Software Engineer: ${calibrated_zero_exp_swe_mean:,.2f}")
    else:
        training_target = TARGET_COLUMN
        print("Using original salary target for training.")

    X = df.drop(columns=[TARGET_COLUMN, CALIBRATED_TARGET_COLUMN], errors="ignore")
    y = df[training_target]

    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    baseline_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_depth=24,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=4,
        ),
        "MLP Regressor": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=150,
            random_state=42,
            early_stopping=True,
        ),
    }

    baseline_results = {}
    trained_baselines = {}

    print("Training baseline models with all features...")
    for model_name, model in baseline_models.items():
        pipeline, metrics = train_and_evaluate_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        baseline_results[model_name] = metrics
        trained_baselines[model_name] = pipeline

    baseline_table = pd.DataFrame(baseline_results).T.sort_values("R2", ascending=False)
    print("\nBaseline comparison:")
    print(baseline_table)

    print("\nRunning Genetic Algorithm for feature selection...")
    selector = GeneticFeatureSelector(
        population_size=30,
        generations=20,
        mutation_rate=0.1,
        crossover_rate=0.8,
        random_state=42,
        model_n_estimators=100,
        max_train_rows=100000,
    )

    ga_result = selector.fit(X_train, y_train)
    selected_features = ga_result.best_features

    print("\nBest GA-selected features:")
    print(selected_features)
    print(f"Best GA fitness: {ga_result.best_fitness:.4f}")

    final_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=26,
    )

    final_pipeline, final_metrics = train_and_evaluate_model(
        "Final GA-selected Random Forest",
        final_model,
        X_train,
        X_test,
        y_train,
        y_test,
        selected_features=selected_features,
    )

    best_baseline_name = baseline_table.index[0]
    best_baseline_metrics = baseline_results[best_baseline_name]

    comparison_table = pd.DataFrame(
        {
            f"Best baseline ({best_baseline_name})": best_baseline_metrics,
            "GA-selected final model": final_metrics,
        }
    ).T

    print("\nFinal comparison:")
    print(comparison_table)

    importance_df = get_feature_importance(final_pipeline, selected_features)
    top_15_importance = importance_df.head(15)
    aggregated_importance = aggregate_importance(importance_df, selected_features)

    print("\nTop 15 transformed feature importances:")
    print(top_15_importance)

    print("\nAggregated original feature importances:")
    print(aggregated_importance)

    model_package = {
        "selected_features": selected_features,
        "pipeline": final_pipeline,
        "metadata": {
            "original_target_column": TARGET_COLUMN,
            "training_target_column": training_target,
            "use_experience_calibrated_target": USE_EXPERIENCE_CALIBRATED_TARGET,
            "all_features": ALL_FEATURES,
            "engineered_features": ["experience_level_group"],
            "target_calibration": {
                "reason": "The original synthetic dataset gives very high salaries to zero-experience candidates, so the MVP trains on a salary target gradually down-weighted for 0-9 years of experience.",
                "factor_formula": "if experience_years >= 10: 1.0, else 0.55 + 0.45 * experience_years / 10",
                "factor_examples": {"0_years": 0.55, "5_years": 0.775, "10_plus_years": 1.0},
            },
            "experience_level_mapping": {
                "Junior": "0-2 years",
                "Mid": "3-5 years",
                "Senior": "6-10 years",
                "Expert": "11+ years",
            },
            "ga_best_fitness": ga_result.best_fitness,
            "ga_history": ga_result.history,
            "final_metrics": final_metrics,
            "best_baseline_name": best_baseline_name,
            "best_baseline_metrics": best_baseline_metrics,
        },
    }

    joblib.dump(model_package, MODEL_PATH)

    results = {
        "use_experience_calibrated_target": USE_EXPERIENCE_CALIBRATED_TARGET,
        "training_target_column": training_target,
        "baseline_results": baseline_results,
        "baseline_comparison": baseline_table.to_dict(orient="index"),
        "ga_selected_features": selected_features,
        "ga_best_fitness": ga_result.best_fitness,
        "final_metrics": final_metrics,
        "final_comparison": comparison_table.to_dict(orient="index"),
        "top_15_feature_importance": top_15_importance.to_dict(orient="records"),
        "aggregated_feature_importance": aggregated_importance.to_dict(orient="records"),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
