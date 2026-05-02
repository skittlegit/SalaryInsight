# SalaryInsight — Employee Attrition & Salary Prediction

An end-to-end machine learning pipeline that predicts **employee attrition** and **future salaries**, then quantifies the **expected financial loss** due to turnover — enabling data-driven HR decision-making.

---

## Overview

This project answers three key HR questions:

| Question | Approach |
|----------|----------|
| Who is likely to leave? | Logistic Regression (L1/L2, GridSearchCV) |
| What will they earn in the future? | Stacking Ensemble (XGBoost + GBR + RidgeCV) |
| What is the financial impact? | `P(Attrition) × Predicted Future Salary` |

---

## Project Structure

```
SalaryInsight/
├── data/                          ← place the Kaggle CSV here
├── results/                       ← all outputs saved here
│   ├── attrition_analysis_results_<timestamp>.csv
│   ├── feature_importance_<timestamp>.csv
│   ├── error_metrics_<timestamp>.json
│   ├── confusion_matrix_<timestamp>.png
│   ├── actual_vs_predicted_<timestamp>.png
│   └── feature_importance_plot.png
├── Salary_insight_code.ipynb      ← main notebook
├── requirements.txt
└── README.md
```

---

## Dataset

Uses the [IBM HR Analytics Attrition & Performance dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) from Kaggle (1,470 employees, 35 features).

Download the CSV and place it at `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`, or update `DATA_PATH` at the top of the notebook.

---

## Installation

```bash
git clone https://github.com/skittlegit/SalaryInsight.git
cd SalaryInsight
pip install -r requirements.txt
```

## Usage

1. Place the dataset CSV in `data/`
2. Open `Salary_insight_code.ipynb`
3. Set `DATA_PATH` in the first cell if needed
4. Run all cells — all outputs are saved to `results/`

---

## Models

**Classification — Logistic Regression**  
Tuned with GridSearchCV (C, L1/L2 penalty), 5-fold CV, scored by F1. Class imbalance handled with SMOTEENN.

**Regression — Stacking Ensemble**  
XGBoost + Gradient Boosting as base learners, RidgeCV as meta-learner. Trained only on employees likely to stay (`P(stay) > 0.70`). Target is log-transformed for stability.

---

## Results

All outputs are saved to `results/` with a timestamp so runs never overwrite each other.

| File | Description |
|------|-------------|
| `attrition_analysis_results_<ts>.csv` | Full processed dataset with predictions |
| `feature_importance_<ts>.csv` | Feature importance scores |
| `error_metrics_<ts>.json` | Train/test metrics for both models |
| `confusion_matrix_<ts>.png` | Confusion matrix heatmap |
| `actual_vs_predicted_<ts>.png` | Actual vs. predicted salary plot |
| `feature_importance_plot.png` | Top features bar chart |

---

## License

MIT
