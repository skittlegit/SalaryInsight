# SalaryInsight — Employee Attrition & Salary Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189AB4)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning pipeline that predicts **employee attrition** and **future salaries**, then quantifies the **expected financial loss** due to turnover — enabling data-driven HR decision-making.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Results & Outputs](#-results--outputs)
- [Key Features](#-key-features)
- [License](#-license)

---

## 🧠 Project Overview

This project answers three critical HR questions by combining **classification** and **regression**:

| # | Question | Approach |
|---|----------|----------|
| 1 | **Who is likely to leave?** | Logistic Regression with GridSearchCV (L1/L2) |
| 2 | **What will they earn in the future?** | Stacking Ensemble (XGBoost + GBR + RidgeCV) |
| 3 | **What is the expected financial impact?** | `P(Attrition) × Predicted Future Salary` |

The expected financial loss formula:

$$\text{Expected Loss} = P(\text{Attrition}) \times \text{Predicted Future Salary}$$

---

## 🔄 Pipeline Architecture

```
Raw Data (IBM HR Dataset)
        │
        ▼
┌─────────────────────────────┐
│  1. Data Loading & Validation│
│     · Schema checks          │
│     · Value range validation │
└──────────────┬──────────────┘
               │
        ▼
┌─────────────────────────────┐
│  2. Preprocessing            │
│     · Label Encoding         │
│     · KNN Imputation         │
└──────────────┬──────────────┘
               │
        ▼
┌─────────────────────────────┐
│  3. Feature Engineering      │
│     · Domain features        │
│     · Polynomial transforms  │
│     · Log transforms         │
└──────────────┬──────────────┘
               │
        ▼
┌─────────────────────────────┐
│  4. Class Imbalance          │
│     · SMOTEENN               │
└──────────────┬──────────────┘
               │
        ▼
┌─────────────────────────────┐
│  5. Classification           │
│     · Logistic Regression    │
│     · GridSearchCV, 5-fold   │
└──────────────┬──────────────┘
               │
        ▼
┌─────────────────────────────┐
│  6. Regression               │
│     · XGBoost + GBR Stack    │
│     · RidgeCV meta-learner   │
│     · Log-target transform   │
└──────────────┬──────────────┘
               │
        ▼
┌─────────────────────────────┐
│  7. Evaluation & Outputs     │
│     · Confusion matrix       │
│     · ROC / AUC curve        │
│     · Residual plots         │
│     · Feature importance     │
└─────────────────────────────┘
```

---

## 📁 Project Structure

```
SalaryInsight/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   ← download from Kaggle
├── Salary_insight_code.ipynb                     ← main notebook
├── requirements.txt
├── README.md
│
│   ── Generated on run ──────────────────────────
├── attrition_analysis_results_<timestamp>.csv
├── feature_importance_<timestamp>.csv
├── error_metrics_<timestamp>.json
├── confusion_matrix_<timestamp>.png
├── roc_curve_<timestamp>.png
├── residual_plot_<timestamp>.png
├── actual_vs_predicted_<timestamp>.png
├── feature_importance_plot.png
├── attrition_classifier_<timestamp>.pkl
├── salary_regressor_<timestamp>.pkl
├── scaler_cls_<timestamp>.pkl
└── scaler_reg_<timestamp>.pkl
```

---

## 📚 Dataset

This project uses the [IBM HR Analytics Attrition & Performance dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) from Kaggle.

| Property | Value |
|----------|-------|
| Rows | 1,470 employees |
| Features | 35 original columns |
| Classification target | `Attrition` (Yes / No) |
| Regression target | Predicted `FutureSalary` |
| Class distribution | ~84 % No · ~16 % Yes |

**Setup:** Download the CSV and place it at:

```
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Or update the `DATA_PATH` variable at the top of the notebook.

---

## 🛠️ Installation

### Prerequisites

- Python 3.9 +
- pip or conda

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/SalaryInsight.git
cd SalaryInsight

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset in the data/ folder

# 5. Launch the notebook
jupyter notebook Salary_insight_code.ipynb
```

---

## 🚀 Usage

Open `Salary_insight_code.ipynb` and either:

**Option A — Run all cells sequentially** (recommended):
1. Set `DATA_PATH` in the configuration cell at the top
2. **Run → Run All Cells**

**Option B — Run the main pipeline function directly** (from the last cell):
```python
if __name__ == "__main__":
    df, metrics, enhanced_models, features = main()
```

---

## 🤖 Model Details

### Classification — Logistic Regression

| Hyperparameter | Search Space |
|---------------|-------------|
| `C` (regularization strength) | 0.01, 0.1, 1, 10 |
| `penalty` | L1, L2 |
| `solver` | liblinear |
| CV strategy | StratifiedKFold, 5 folds |
| Scoring metric | F1 |

Class imbalance is handled with **SMOTEENN** (combined synthetic over-sampling + Edited Nearest Neighbours under-sampling).

### Regression — Stacking Ensemble

| Component | Algorithm |
|-----------|-----------|
| Base learner 1 | XGBoost Regressor (tuned via GridSearchCV) |
| Base learner 2 | Gradient Boosting Regressor |
| Meta-learner | RidgeCV |
| Target transformation | `log1p(FutureSalary)` |
| CV folds | 3 |

**Future salary** is simulated with a performance-based increment:
- Rating 4 → **+10 %**
- Rating 1–3 → **+5 %**

---

## 📊 Results & Outputs

All output files are timestamped to prevent overwrites:

| File | Description |
|------|-------------|
| `attrition_analysis_results_<ts>.csv` | Full processed dataset with predictions |
| `feature_importance_<ts>.csv` | Ranked feature importance scores |
| `error_metrics_<ts>.json` | Classification & regression metrics (train/test) |
| `confusion_matrix_<ts>.png` | Classification confusion matrix heatmap |
| `roc_curve_<ts>.png` | ROC curve with AUC score |
| `residual_plot_<ts>.png` | Regression residual scatter plot |
| `actual_vs_predicted_<ts>.png` | Predicted vs. actual salary plot |
| `feature_importance_plot.png` | Top feature importance bar chart |
| `*.pkl` | Serialized model and scaler artifacts |

---

## ✨ Key Features

- **Robust data validation** — checks schema, column types, and value ranges on load
- **KNN Imputation** — handles missing values preserving distributional structure
- **10+ engineered features** — domain-specific ratios (income/age, career progress, promotion rate, etc.)
- **Polynomial features** — captures non-linear income × tenure relationships
- **SMOTEENN** — corrects class imbalance without discarding majority-class data
- **Stacking ensemble** — combines XGBoost and GBR under a Ridge meta-learner
- **Timestamped artifacts** — every run produces versioned outputs
- **Structured logging** — full pipeline traceability with timestamps

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
