# Job Salary Prediction MVP

## Project title

**Job Salary Prediction based on Resume / Job Profile Data**

This project is an academic MVP for the university subject **Biologically Inspired Artificial Intelligence**.

## Problem statement

The goal is to predict a candidate's or job profile's expected salary using structured resume/job-related features. The model takes information such as job title, experience, education, skills, industry, company size, work location, remote work format and certifications, then returns a numeric salary prediction.

## Dataset description

The dataset is stored in:

```text
data/job_salary_prediction_dataset.csv
```

Dataset properties:

- 250,000 rows
- 10 original columns
- 1 engineered feature added during preprocessing: `experience_level_group`
- no missing values
- no duplicate rows

## Features and target

### Input features

Categorical features:

- `job_title`
- `education_level`
- `industry`
- `company_size`
- `location`
- `remote_work`
- `experience_level_group` *(engineered from `experience_years`: Junior, Mid, Senior, Expert)*

Numeric features:

- `experience_years`
- `skills_count`
- `certifications`
- `experience_level_group`

### Target variable

By default, training uses an experience-calibrated target:

- original CSV target: `salary`
- training target created in preprocessing: `salary_calibrated`

This was added because the original synthetic dataset assigns unrealistically high salaries to some zero-experience candidates. For example, a 0-year Software Engineer profile can easily lead to predictions above 100k USD if the model is trained directly on the raw `salary` column.

The calibration does not overwrite the CSV. In `src/train_model.py`, it can be disabled by setting:

```python
USE_EXPERIENCE_CALIBRATED_TARGET = False
```


## Feature engineering: experience levels

The project now derives an additional categorical feature from `experience_years`:

| Experience years | Derived level |
|---:|---|
| 0-2 | `Junior` |
| 3-5 | `Mid` |
| 6-10 | `Senior` |
| 11+ | `Expert` |

The original numeric `experience_years` column is still used. The engineered feature gives the model a second, more interpretable seniority signal. The Genetic Algorithm can decide whether this feature is useful and include or exclude it from the final feature subset.

## Salary target calibration

The original dataset is convenient for a clean ML MVP, but it is not fully realistic. Junior profiles can have very high target salaries. To make the Streamlit demo more believable, the project can train on a calibrated target:

```text
if experience_years >= 10:
    factor = 1.0
else:
    factor = 0.55 + 0.45 * experience_years / 10

salary_calibrated = salary * factor
```

Examples:

| Experience years | Factor | Effect |
|---:|---:|---|
| 0 | 0.55 | strong junior correction |
| 5 | 0.775 | moderate correction |
| 10+ | 1.00 | no correction |

This keeps the dataset structure unchanged but makes predictions for entry-level candidates more realistic.

## Why salary prediction is useful

Salary prediction can help with:

- compensation benchmarking;
- career planning;
- job market analysis;
- estimating the impact of experience, education and skills;
- building educational demos of regression systems.

This project is not intended for real HR decision-making.

## Biologically inspired AI component

The biologically inspired component is a **Genetic Algorithm for feature selection**.

Genetic Algorithms are inspired by biological evolution. They use concepts such as:

- population;
- individual;
- gene;
- fitness;
- selection;
- crossover;
- mutation;
- survival of better solutions.

## Genetic Algorithm explanation

In this project:

- each individual is a binary vector of length 10;
- each gene represents one original or engineered feature group;
- `1` means the feature is selected;
- `0` means the feature is not selected.

Example:

```text
[1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
```

This means that the selected features are:

- `job_title`
- `experience_years`
- `skills_count`
- `company_size`
- `location`
- `certifications`

The fitness function trains a Random Forest Regressor on selected features and evaluates it using validation R² score. A small penalty is added for selecting too many features, encouraging simpler models when scores are similar.

Default GA parameters after the longer-training update:

- `population_size = 30`
- `generations = 20`
- `mutation_rate = 0.1`
- `crossover_rate = 0.8`
- `model_n_estimators = 100` inside GA fitness evaluation
- `max_train_rows = 100000` sampled rows for GA evaluation
- `random_state = 42`

## Machine learning pipeline

The pipeline contains:

1. data loading;
2. exploratory data analysis;
3. train/test split;
4. preprocessing with `ColumnTransformer`;
5. feature engineering: `experience_level_group`;
6. baseline model training;
7. Genetic Algorithm feature selection;
8. final model training;
9. model evaluation on the selected training target;
10. feature importance analysis;
11. saving final model package with `joblib`;
12. Streamlit prediction app.

## Models compared

Baseline models trained with all features:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- MLP Regressor

Final model:

- Random Forest Regressor trained on the GA-selected feature subset

## Evaluation metrics

The project uses:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² — coefficient of determination

## Final results table

After running the training script, results are saved to:

```text
models/training_results.json
```

The script prints:

- baseline comparison table;
- GA progress by generation;
- selected feature subset;
- final model metrics;
- comparison between the best baseline and GA-selected final model;
- top feature importances.

## How to run the project

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models:

```bash
python src/train_model.py
```

The updated configuration trains longer than the initial MVP version. On slower laptops, the GA stage can take noticeable time because it evaluates 30 individuals for 20 generations using Random Forest models.

The trained model will be saved to:

```text
models/salary_model.pkl
```

## How to run Streamlit app

After training the model, run:

```bash
streamlit run app.py
```

The app allows the user to enter a structured job/resume profile and predicts salary.

## Project structure

```text
job-salary-prediction-mvp/
├── data/
│   └── job_salary_prediction_dataset.csv
├── notebooks/
│   └── 01_eda_and_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── genetic_feature_selection.py
│   ├── train_model.py
│   └── predict.py
├── models/
├── app.py
├── requirements.txt
└── README.md
```

## Limitations

- The dataset contains structured profile data, not full resume text.
- No NLP or text embeddings are used.
- The model learns patterns from the provided dataset only.
- The original raw target is synthetic-like and can produce unrealistic junior salaries.
- Salary prediction may reflect synthetic or historical biases in data.
- The MVP is designed for academic demonstration, not real HR use.
- Genetic Algorithm feature selection is useful for demonstrating bio-inspired optimization, but it can be computationally expensive.

## Future improvements

Possible extensions:

- add cross-validation inside the GA fitness function;
- compare more evolutionary strategies;
- tune model hyperparameters with GA;
- add SHAP explainability;
- use a larger and more realistic real-world salary dataset;
- add fairness and bias checks;
- add confidence intervals for predictions;
- deploy the app with Docker;
- add automated tests and CI.
