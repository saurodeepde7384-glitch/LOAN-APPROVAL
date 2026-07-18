# Loan Approval Prediction using Machine Learning

A complete machine learning project that predicts whether a loan application will be approved or rejected based on applicant details such as income, education, employment status, CIBIL score, loan amount, assets, and other financial indicators.

The project demonstrates an end-to-end ML workflow including data cleaning, exploratory data analysis (EDA), preprocessing with scikit-learn Pipelines, model training, evaluation, comparison of multiple algorithms, and model persistence using Joblib.

---

## Project Overview

This project aims to build a reliable classification model for loan approval prediction. The dataset contains information about applicants and their financial background. Multiple machine learning models were trained and evaluated, and Random Forest was selected as the final model based on its superior performance.

### Problem Statement

Financial institutions need to determine whether a loan applicant is likely to be approved. This project automates that decision-making process using historical applicant data and machine learning classification algorithms.

---

## Dataset Features

The dataset includes the following features:

* `loan_id`
* `no_of_dependents`
* `education`
* `self_employed`
* `income_annum`
* `loan_amount`
* `loan_term`
* `cibil_score`
* `residential_assets_value`
* `commercial_assets_value`
* `luxury_assets_value`
* `bank_asset_value`
* `loan_status` (Target variable)

---

## Project Workflow

1. Data loading and inspection
2. Data cleaning

   * Removed duplicates
   * Stripped extra spaces from categorical values
   * Handled missing values
3. Exploratory Data Analysis (EDA)
4. Train-test split
5. Data preprocessing using `Pipeline` and `ColumnTransformer`

   * `SimpleImputer`
   * `StandardScaler`
   * `OneHotEncoder`
6. Model training
7. Model evaluation and comparison
8. Model selection
9. Model saving using Joblib

---

## Exploratory Data Analysis (EDA)

EDA was performed in a separate notebook to understand:

* Distribution of loan approvals and rejections
* Income distribution
* Loan amount distribution
* CIBIL score distribution
* Relationship between income and loan amount
* Asset value distributions
* Correlation between numerical features
* Outlier detection using boxplots

---

## Preprocessing Pipeline

A professional scikit-learn preprocessing pipeline was used to avoid data leakage and ensure consistent transformations during both training and prediction.

### Numerical Features

* Missing value imputation using `SimpleImputer(strategy="mean")`
* Feature scaling using `StandardScaler()`

### Categorical Features

* Missing value imputation using `SimpleImputer(strategy="most_frequent")`
* Encoding using `OneHotEncoder(handle_unknown="ignore")`

### Pipeline Structure

```
Raw Data
   ↓
ColumnTransformer
   ├── Numeric Pipeline
   │      ├── SimpleImputer
   │      └── StandardScaler
   └── Categorical Pipeline
          ├── SimpleImputer
          └── OneHotEncoder
   ↓
Random Forest Classifier
```

---

## Models Trained

The following classification models were trained and evaluated:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* Gradient Boosting

---

## Model Evaluation

The models were evaluated using Accuracy, Precision, Recall, and F1 Score.

| Model               | Accuracy   | Precision  | Recall     | F1 Score   |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.9024     | 0.8650     | 0.8705     | 0.8677     |
| KNN                 | 0.8821     | 0.8200     | 0.8705     | 0.8445     |
| SVM                 | 0.9266     | 0.8887     | 0.9151     | 0.9017     |
| Decision Tree       | 0.9774     | 0.9784     | 0.9689     | 0.9689     |
| **Random Forest**   | **0.9805** | **0.9785** | **0.9682** | **0.9733** |
| Gradient Boosting   | 0.9750     | 0.9803     | 0.9512     | 0.9655     |

---

## Final Model Selection

**Random Forest Classifier** was selected as the final model because it achieved the best overall performance across all evaluation metrics.

### Final Model Performance

* **Accuracy:** 98.05%
* **Precision:** 97.85%
* **Recall:** 96.82%
* **F1 Score:** 97.33%

The model provides a strong balance between precision and recall, making it suitable for loan approval prediction tasks.

---

## Model Persistence

The complete trained pipeline, including preprocessing steps and the Random Forest model, was saved using Joblib.

```python
import joblib

joblib.dump(model, "model/Loan_approval_pipeline.pkl")
```

To load the saved model:

```python
model = joblib.load("model/Loan_approval_pipeline.pkl")
```

This ensures that new input data will undergo the same preprocessing before prediction.

---

## Project Structure

```
LOAN-APPROVAL/
│
├── data/
│   └── loan_approval_dataset.csv
│
├── notebooks/
│   ├── Loan_approval_eda.ipynb
│   └── Loan_approval_training.ipynb
│
├── model/
│   └── Loan_approval_pipeline.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Jupyter Notebook

---

## How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/saurodeepde7384-glitch/LOAN-APPROVAL.git
cd LOAN-APPROVAL
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the notebooks

* `Loan_approval_eda.ipynb` for exploratory data analysis
* `Loan_approval_training.ipynb` for model training and evaluation

4. Load the saved model

```python
import joblib

model = joblib.load("model/Loan_approval_pipeline.pkl")
```

---

## Future Improvements

* Add cross-validation
* Perform hyperparameter tuning using `GridSearchCV`
* Build a Streamlit web application
* Deploy the application online
* Add feature importance visualization
* Include confusion matrix and ROC curve analysis

---

## Learning Outcomes

Through this project, I learned:

* Data cleaning and preprocessing
* Handling missing values
* Encoding categorical variables
* Feature scaling
* Building scikit-learn Pipelines
* Using ColumnTransformer
* Training multiple classification models
* Evaluating models using multiple metrics
* Saving and loading trained models with Joblib
* Organizing a professional ML project structure

---

## Author

**Saurodeep De**
> This project was built as part of my machine learning learning journey and portfolio development.

---

## License

> This project is open-source and free to use for educational purpose.
