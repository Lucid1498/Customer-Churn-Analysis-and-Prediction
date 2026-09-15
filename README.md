# Customer Churn Analysis and Prediction

An end-to-end data science project focused on understanding customer churn patterns and building machine learning models to identify customers at risk of leaving a telecommunications provider.

The project combines exploratory data analysis, feature-level churn investigation, class-imbalance handling with SMOTE, model comparison, and hyperparameter tuning.

## Project Overview

Customer churn is an important business problem because losing existing customers can directly affect revenue and customer lifetime value.

This project analyzes a telecommunications customer dataset to answer two main questions:

1. Which customer characteristics are most strongly associated with churn?
2. How effectively can machine learning models identify customers who are likely to churn?

The project is organized into two stages:

- **Exploratory Data Analysis**
- **Predictive Modeling**

## Dataset

The dataset contains customer-level information for a telecommunications company.

Key features include:

- Customer tenure
- Monthly charges
- Total charges
- Contract type
- Internet service
- Payment method
- Online security
- Technical support
- Device protection
- Demographic and household information
- Customer churn status

After cleaning, the dataset contains approximately 7,000 customer records.

## Repository Structure

```text
Customer-Churn-Analysis-and-Prediction/
│
├── data/
│   └── telco-customer-churn.csv
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_churn_prediction_modeling.ipynb
│
├── reports/
│   └── comparison_report.html
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Exploratory Data Analysis

The first notebook focuses on data quality, churn patterns, and feature relationships.

Main steps include:

- Data inspection and cleaning
- Conversion and validation of `TotalCharges`
- Missing-value handling
- Categorical encoding
- Outlier inspection
- Target-class distribution analysis
- Numerical feature analysis
- Contract and payment-method analysis
- Service-related churn analysis
- Correlation analysis
- Exploratory feature importance
- Stratified train/test comparison

### Key EDA Findings

Approximately **26.6% of customers churned**, while 73.4% remained.

Several strong churn patterns were identified:

- Customers with shorter tenure were considerably more likely to churn.
- Month-to-month customers had a churn rate of approximately **42.7%**, compared with:
  - **11.3%** for one-year contracts
  - **2.8%** for two-year contracts
- Customers using electronic check had a churn rate of approximately **45.3%**.
- Fiber optic customers showed substantially higher churn than DSL customers.
- Customers without online security or technical support showed higher churn rates.
- Churned customers tended to have higher monthly charges.

Exploratory Decision Tree feature importance highlighted:

- Tenure
- Total Charges
- Monthly Charges
- Fiber optic internet service

as some of the most informative variables.

## Machine Learning Workflow

The second notebook builds classification models using the cleaned and encoded dataset.

The following algorithms were evaluated:

- Gaussian Naive Bayes
- Logistic Regression
- Random Forest
- XGBoost

The workflow includes:

1. Train/test split using stratification
2. Feature scaling where appropriate
3. Baseline model evaluation
4. Confusion-matrix analysis
5. Class balancing with SMOTE
6. Pipeline-based resampling
7. Hyperparameter tuning with cross-validation
8. Final test-set evaluation

## Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Recall was treated as particularly important because false negatives represent customers who actually churned but were not identified in advance.

## Baseline Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Naive Bayes | 0.645 | 0.418 | 0.864 | 0.564 | 0.810 |
| Logistic Regression | 0.804 | 0.648 | 0.575 | 0.609 | 0.836 |
| Random Forest | 0.785 | 0.619 | 0.500 | 0.553 | 0.815 |
| XGBoost | 0.767 | 0.568 | 0.516 | 0.541 | 0.814 |

Logistic Regression produced the strongest overall baseline performance, while Naive Bayes achieved the highest recall at the cost of many false-positive predictions.

## Effect of SMOTE

SMOTE was applied through machine-learning pipelines so that synthetic samples were generated only during model training.

The largest recall improvements were:

- Logistic Regression: **0.575 → 0.781**
- Random Forest: **0.500 → 0.599**
- XGBoost: **0.516 → 0.580**

SMOTE improved churn detection but generally reduced precision and accuracy, highlighting the tradeoff between detecting more churners and generating additional false positives.

## Hyperparameter Tuning

Hyperparameter tuning was performed using stratified cross-validation and SMOTE-enabled pipelines.

Best cross-validation F1 scores:

| Model | Best CV F1 |
|---|---:|
| Logistic Regression | 0.631 |
| Random Forest | 0.623 |
| XGBoost | 0.626 |

## Final Tuned Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.736 | 0.502 | 0.775 | 0.609 | 0.834 |
| Random Forest | 0.767 | 0.549 | 0.684 | 0.610 | 0.826 |
| XGBoost | 0.750 | 0.522 | 0.698 | 0.597 | 0.824 |

## Model Selection

The tuned **Random Forest** model provides the strongest overall balance between precision and recall.

It achieves:

- Accuracy: **76.7%**
- Precision: **54.9%**
- Recall: **68.4%**
- F1-score: **0.610**
- ROC-AUC: **0.826**

However, if the primary business objective is to identify as many potential churners as possible, tuned Logistic Regression remains a strong alternative because of its higher recall of **77.5%**.

This illustrates an important practical point: model selection should depend on the business cost of false negatives compared with false positives.

## Key Takeaways

This project demonstrates that:

- Customer tenure and contract type are strongly associated with churn.
- Billing and service-related features provide useful predictive information.
- Accuracy alone is not sufficient for evaluating churn models.
- SMOTE can improve minority-class recall, but usually introduces a precision tradeoff.
- Hyperparameter tuning substantially improved the tree-based models.
- The best model depends on the operational objective of the retention strategy.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn
- XGBoost
- SweetViz
- Jupyter Notebook

## Running the Project

Clone the repository:

```bash
git clone https://github.com/Lucid1498/Customer-Churn-Analysis-and-Prediction.git
```

Navigate into the project:

```bash
cd Customer-Churn-Analysis-and-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebooks in Jupyter Notebook, JupyterLab, or VS Code.

Recommended order:

```text
01_exploratory_data_analysis.ipynb
02_churn_prediction_modeling.ipynb
```

## Project Background

This project originated as a two-part graduate course project for **CS 675 – Introduction to Data Science** at Pace University.

It was later revisited and refactored to improve:

- Project organization
- Reproducibility
- Data preprocessing
- Model evaluation
- Class-imbalance handling
- Hyperparameter tuning
- Documentation
- Portfolio presentation

The updated repository preserves the original analytical objective while applying improved data science practices.
