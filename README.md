# 🚆 Train Delay Prediction Using R

Predicting whether an Indian Railways journey will be delayed, using an end-to-end R workflow: data understanding, cleaning, preprocessing, EDA, feature engineering, PCA and classification models.

> **Course:** Data Science & Analytics (DSA)
> **Institute:** Government College of Engineering Kalahandi, Dept. of Computer Science & Engineering


---

## 👥 Team & Contributions

| Member | Contribution |
|---|---|
| **Barsha Ranee Lenka** | Data understanding and cleaning |
| **Ankita Hota** | Data preprocessing and EDA |
| **Palin Panigrahi** | Feature engineering and feature extraction (PCA) |
| **Prerana Mishra** | Predictive modelling and verification |

---

## 🎯 Problem Statement

Given details of a train journey (route, station categories, train and traction type, weather and operational risk, departure date, etc.), predict the target **`is_delayed`**:

- `0` → journey not delayed
- `1` → journey delayed (by more than 15 minutes)

**Dataset:** Indian Railways Train Delay Dataset (`ir_train.csv`, `ir_test.csv`)

---

## 🔄 Workflow

1. **Data understanding:** identify the target, ID column and date field. `delay_minutes` and `primary_delay_cause` are excluded because they describe the outcome itself (data leakage).
2. **Data cleaning:** remove identifiers and leaky columns, derive year/month/day/week from `departure_date`.
3. **Preprocessing:** median imputation for numeric values, mode imputation for categorical values, outlier capping at the 1st/99th percentiles, and missingness flags. All rules are learned on training data only.
4. **EDA:** target distribution, histograms, correlations, late rates by category, season and zone, and time patterns.
5. **Feature engineering:** station-rank mapping (A1=6 … E=1), `station_category_gap`, `distance_per_hour`, risk features, and sine/cosine encodings of hour and month.
6. **Feature extraction (PCA):** standardised numeric features reduced to **32 principal components** (95% cumulative variance).
7. **Modelling and verification:** Logistic Regression vs Random Forest on an 80/20 train-validation split, evaluated with a confusion matrix and ROC-AUC.

---

## 📊 Key Results

The target is imbalanced: about **71.7% not delayed** and **28.3% delayed** in the EDA sample.

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8360 | 0.8284 | 0.8555 | 0.9362 | 0.8790 | **0.9229** |
| Random Forest | **0.8445** | **0.9347** | 0.6139 | 0.8609 | **0.8963** | 0.9031 |

**Takeaways**

- Random Forest is the better choice when the goal is to catch delayed journeys (highest sensitivity and F1).
- Logistic Regression is a simpler, more specific baseline with the higher ROC-AUC.
- The top Random Forest features are **PC1, PC2, season and PC4**, so the predictive signal is spread across several components rather than one raw variable.

---

## 📁 Repository Structure

```
├── indian-railways-predict-train-delay/   # Dataset files
├── 02_preprocessing_eda.R                 # Preprocessing and EDA
├── 03_predictive_modelling.R              # Feature engineering, PCA and models
├── check_data.R                           # Data checks
├── model_comparison_graph.R               # Model comparison plot
├── plots/                                 # EDA and result visualisations
├── model_results/                         # Saved model outputs
├── final_predictions.csv                  # Final predictions
├── continuous_summary.csv                 # Summary of continuous variables
└── correlation_with_target.csv            # Correlation with the target
```

---

## ▶️ How to Run

1. Install [R](https://cran.r-project.org/) (and optionally RStudio / VS Code with the R extension).
2. Clone this repository:
   ```bash
   git clone https://github.com/Ankitahota22/Train_Delay_Prediction_Using_R.git
   ```
3. Install the required packages:
   ```r
   install.packages(c("dplyr", "tidyr", "ggplot2", "lubridate",
                      "caret", "ranger", "pROC"))
   ```
4. Run the scripts in order:
   ```r
   source("02_preprocessing_eda.R")
   source("03_predictive_modelling.R")
   ```

> Note: file paths and package lists may vary slightly depending on the scripts. Check the top of each `.R` file.

---

## 🛠️ Tech Stack

R · tidyverse (dplyr, tidyr, ggplot2) · lubridate · caret · ranger · pROC

---

