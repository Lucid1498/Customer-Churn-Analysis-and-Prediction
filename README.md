# Customer-Churn-Analysis-and-Prediction
This repository contains projects focusing on Customer Churn Analysis in the telecommunications industry, performed as part of the CS675 - Introduction to Data Science course at Pace University. The work spans exploratory data analysis (EDA) and machine learning model building to identify and predict customer churn.

# Table of Contents
Project Description

Technologies Used

Repository Structure

Instructions to Run

Results and Insights

License

# Project Description

# Project 1: Exploratory Data Analysis (EDA)

Objective: Perform a detailed EDA on customer churn data to uncover trends, correlations, and insights that can guide predictive modeling.

Key Tasks:
    Identify and handle missing values, outliers, and data transformation requirements.
    Use classic EDA libraries like Pandas, NumPy, Matplotlib, and Seaborn to explore the data.
    Generate automated reports using Pandas Profiling and SweetViz for insights.
    Highlight critical features influencing churn.

# Project 2: Machine Learning for Customer Churn Prediction

Objective: Build and evaluate machine learning models to predict customer churn using the dataset analyzed in Project 1.

Key Tasks:
Preprocess the data, including feature encoding and scaling.
Handle class imbalance using SMOTE.
Implement machine learning models (Naïve Bayes, Logistic Regression, Random Forest, XGBoost).
Evaluate models using metrics like Accuracy, Precision, Recall, and F1-Score.
Perform hyperparameter tuning for improved performance.

# Technologies Used:

Languages: Python
Libraries:
    EDA: Pandas, NumPy, Matplotlib, Seaborn, SweetViz, Pandas Profiling
    Machine Learning: Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)


# Run Jupyter Notebooks:

For EDA: Open project1_eda.ipynb in Jupyter Notebook or Jupyter Lab.
For ML Modeling: Open project2_ml_modeling.ipynb.

# Results and Insights:

EDA (Project 1):

Identified key features affecting churn:
Contract Type, Tenure, Monthly Charges, and Internet Service Type.
Correlation analysis and visualizations revealed strong relationships among customer demographics, services, and churn.

ML Modeling (Project 2):

Models achieved high recall on the SMOTE-balanced dataset, critical for minimizing false negatives in churn predictions.
Best Model: Random Forest with Recall: 0.89, Precision: 0.85, F1-Score: 0.87.



