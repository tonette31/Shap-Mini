# SHAP Mini: Explainable AI for Tabular Models

**SHAP Mini** is a lightweight and reproducible project that demonstrates **model explainability** using the **SHAP (SHapley Additive exPlanations)** framework.  
It helps visualize how individual features contribute to model predictions in simple tabular machine learning problems.

The project is intentionally minimal, using only **RandomForest** and **LogisticRegression**, so users can easily inspect, understand, and visualize how SHAP values reveal the inner workings of black-box models.

---

## Folder Structure

```

shap-mini/
│
├── data/                     # Input data folder
│   └── train.csv             # Optional user dataset (auto-generated if missing)
│
├── models/                   # Trained model outputs
│   └── rf.pkl                # Saved RandomForest model
│
├── outputs/                  # All result artifacts
│   ├── metrics.json          # Model performance metrics
│   ├── shap_summary.png      # Global SHAP importance (Figure 1)
│   ├── shap_dependence_feature_0.png  # Dependence for feature_0 (Figure 2)
│   ├── shap_dependence_feature_3.png  # Dependence for feature_3 (Figure 3)
│   └── train_columns.json    # Feature column list for reproducibility
│
├── utils.py                  # JSON + column utility functions
├── train.py                  # Model training script
├── shapify.py                # SHAP explanation script
├── config.yaml               # Simple experiment configuration
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation (this file)

````

---

## Key Features

- **Automatic data generation** if no dataset is provided.
- **Two baseline models:**
  - RandomForestClassifier (`rf`)
  - LogisticRegression (`logreg`)
- **SHAP visualization suite:**
  - Global feature importance bar plot
  - Dependence plots for any feature
- **JSON-based outputs** for reproducibility
- Works on **CPU-only** machines and installs in <1 minute.

---

## Setup Instructions

### Create and activate environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
````

### Train the model

```bash
python train.py --model rf
```

If no `data/train.csv` is present, a **synthetic binary classification dataset** will be created automatically.

This saves:

* `models/rf.pkl`
* `outputs/metrics.json`
* `outputs/train_columns.json`

### Generate SHAP plots

```bash
python shapify.py --model rf --feature feature_3
```

All generated plots are saved under `outputs/`.

---

## Model Performance

| Metric       | Value |
| ------------ | ----- |
| **Accuracy** | 0.93  |
| **F1 Score** | 0.93  |

> *Source: `outputs/metrics.json` (synthetic dataset with 2000 samples, 12 features).*

This confirms the RandomForest learned a strong signal, mainly dominated by two features (`feature_3` and `feature_8`).

---

## Theoretical Background

**SHAP (SHapley Additive Explanations)** assigns each feature an importance value for a particular prediction.
It is based on the concept of **Shapley values** from cooperative game theory.

For each prediction:

* Every feature is treated as a **player** in a coalition game.
* The SHAP value represents the **average marginal contribution** of that feature to the model output across all possible feature subsets.

This yields:

* **Global interpretability** → average impact over all samples.
* **Local interpretability** → explanation of a single instance.

### Benefits of SHAP:

* Model-agnostic and consistent with human reasoning.
* Captures **feature interactions** and **directionality**.
* Provides unified visualization across model types.

---

## Dataset and Model

### Data

* **Synthetic binary classification** with 12 numeric features.
* 3 features are informative (`feature_3`, `feature_8`, `feature_11`).
* 2 redundant and 7 noise variables.
* Split: 80% training / 20% testing.

### Model

* **RandomForestClassifier**

  * 200 estimators
  * Default depth
  * Random seed = 42
* **Objective:** predict the binary target using all 12 features.

---

## SHAP Visualizations and Interpretation

### Figure 1: Global Feature Importance

<img width="800" height="630" alt="shap_summary" src="https://github.com/user-attachments/assets/591cc098-7d00-44d5-947f-a24338194192" />

**What it shows:**
Each bar represents the **mean(|SHAP value|)**, the average magnitude of feature impact on model output.

**Insights:**

* `feature_3` dominates, followed by `feature_8` and `feature_11`.
* Features like `feature_0`, `feature_7`, and `feature_10` have minimal contribution.
* This aligns with the data generation process, where `feature_3` carries the main signal.

**Interpretation:**
The model primarily bases its decisions on a single strong predictor (`feature_3`).
Such concentration can be desirable for interpretability but risky for overfitting if that feature is noisy.

---

### Figure 2: Dependence Plot for `feature_0`

<img width="750" height="500" alt="shap_dependence_feature_0" src="https://github.com/user-attachments/assets/ea49d8c5-98af-4259-abb9-1b641c8f0887" />

**What it shows:**
A scatter of SHAP values vs feature values, colored by interaction with another feature (`feature_3`).

**Insights:**

* SHAP values cluster near 0 → `feature_0` has almost no effect.
* The color gradient (based on `feature_3`) shows **minimal interaction**.
* This indicates `feature_0` contributes little independent or joint information.

**Interpretation:**
This feature could be dropped without affecting model accuracy, simplifying the model further.

---

### Figure 3: Dependence Plot for `feature_3`

<img width="750" height="500" alt="shap_dependence_feature_3" src="https://github.com/user-attachments/assets/389def70-1fa2-40a0-a672-5215add3f56e" />

**What it shows:**
The dominant feature’s SHAP values plotted against its values, colored by `feature_8`.

**Insights:**

* A strong linear relationship → higher `feature_3` → higher SHAP contribution → higher predicted probability.
* The plot forms a near-perfect diagonal, confirming **monotonic and consistent influence**.
* Color (`feature_8`) reveals a weak secondary interaction: samples with high `feature_8` intensify the effect of `feature_3`.

**Interpretation:**
`feature_3` is the **main decision axis** in the model.
It behaves predictably and explains most of the model variance, making it an excellent candidate for feature-based decision rules or business logic extraction.

---

## Output Files Explained

| File                                    | Description                        |
| --------------------------------------- | ---------------------------------- |
| `models/rf.pkl`                         | Trained RandomForest model         |
| `outputs/metrics.json`                  | Performance metrics (accuracy, f1) |
| `outputs/shap_summary.png`              | Global importance (Figure 1)       |
| `outputs/shap_dependence_feature_0.png` | Low-impact feature plot (Figure 2) |
| `outputs/shap_dependence_feature_3.png` | Dominant feature plot (Figure 3)   |
| `outputs/train_columns.json`            | Feature order used during training |

---

## Reproducibility

To reproduce results exactly:

```bash
python train.py --model rf --seed 42
python shapify.py --model rf --feature feature_3
```

Environment:

```
Python >= 3.10
numpy >= 1.21
pandas >= 1.3
matplotlib >= 3.4
scikit-learn >= 1.0
shap >= 0.45
```

All random seeds are fixed (`random_state=42`) to ensure deterministic splits and SHAP outcomes.

---

## Deep Dive: Why SHAP?

Unlike black-box accuracy metrics, SHAP values let you:

* Quantify *how much* each variable pushes a prediction up or down.
* Compare **global importance** and **local influence** simultaneously.
* Audit ML systems for fairness, drift, and bias.

**In this project:**

* SHAP revealed that even with multiple correlated features, one (`feature_3`) dominates.
* Such insight can guide feature selection, data collection, or domain validation.

---

## Extending This Project

You can easily extend **SHAP Mini** to support:

1. **Other models:**

   * `XGBoost`, `LightGBM`, or `CatBoost`
2. **Regression tasks:**

   * Replace classifier with `RandomForestRegressor`
3. **Custom datasets:**

   * Place your own `data/train.csv` (must include `target` column)
4. **Interactive dashboards:**

   * Use `streamlit` or `gradio` to visualize SHAP interactively.

Example:

```bash
python shapify.py --model rf --feature feature_8
```

---

## Conclusion

This project demonstrates that:

* SHAP can make complex models interpretable.
* Even a simple RandomForest can show clear, explainable feature effects.
* Combining **global** and **local** SHAP views provides a holistic understanding of model behavior.

The figures you generated are not just visuals, they tell a **story of how the model thinks**.

* **Figure 1:** What features matter overall
* **Figure 2:** What doesn’t matter
* **Figure 3:** Why the model succeeds

Together, they form a complete interpretability workflow for small tabular models.
