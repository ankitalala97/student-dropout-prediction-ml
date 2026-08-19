# Student Dropout Prediction Using Machine Learning

## Project Overview
This project focuses on predicting student dropout risk using machine learning. The goal is to identify students who may be at risk of dropping out early enough for academic advisors or support teams to intervene.

The project uses the Student Dropout and Academic Success dataset from the UCI Machine Learning Repository. The dataset contains academic performance, financial status, demographic, and macroeconomic variables. The technical report describes the dataset as having 4,424 records and 36 features.

## Business Problem
Student dropout affects students, institutions, and the broader economy. Students may lose career and income benefits from completing a degree, while institutions may lose tuition revenue and graduation/completion performance.

The goal of this project is to create an early-warning model that can flag at-risk students so support teams can take action before the student fully disengages.

## My Contribution
This was an academic machine learning project. My contribution focused on understanding the business problem, supporting machine learning analysis, interpreting model results, and helping document the final technical report and business recommendations.

## Dataset
- Source: UCI Machine Learning Repository
- Dataset: Student Dropout and Academic Success
- Records: 4,424
- Features: 36
- Target: Student outcome converted into binary classification

The original target had three classes:
- Graduate
- Dropout
- Enrolled

For this project, the target was converted into a binary outcome:

- Dropout = 1
- Non-Dropout = 0

Graduate and Enrolled students were grouped together as Non-Dropout because the business objective was to identify students at risk of leaving.

## Machine Learning Workflow
The project follows an end-to-end machine learning pipeline:

1. Data loading and inspection
2. Data preprocessing
3. Binary target creation
4. Low-signal column removal
5. Outlier treatment using Winsorization
6. Feature engineering
7. Class imbalance handling using random oversampling
8. Train/test split
9. Feature scaling using StandardScaler
10. Base model training
11. Hyperparameter tuning using GridSearchCV
12. Model comparison and selection
13. Feature importance analysis
14. Threshold analysis
15. Business interpretation and impact analysis

## Feature Engineering
Four domain-informed features were created from academic performance variables:

- `sem1_approval_rate`
- `sem2_approval_rate`
- `total_approved`
- `avg_grade`

These features were designed to better capture student academic progress and engagement.

## Models Evaluated
The project compared multiple machine learning models:

- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- Gradient Boosting

Gradient Boosting was selected as the recommended model based on model performance, including ROC-AUC and F1-score.

## Why Recall Matters
For dropout prediction, recall is especially important because missing a truly at-risk student is more costly than incorrectly flagging a student who is not actually at risk.

A false negative means the model fails to identify a student who may drop out, causing the student to miss potential support or intervention. Because of this, the project included threshold analysis to support better recall for early-warning use cases.

## Files in This Repository
- `dropout_predictor.py` — Python script containing the end-to-end ML pipeline
- `student_dropout_prediction_notebook.ipynb` — Jupyter Notebook version of the project
- `students_dropout_academic_success.csv` — dataset used for model development
- `docs/Student_Dropout_Prediction_Technical_Report.pdf` — full technical report

## Tools and Libraries Used
- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook
- GridSearchCV
- StandardScaler
- Random oversampling
- Machine learning model evaluation

## Skills Demonstrated
- Machine learning pipeline development
- Data preprocessing
- Feature engineering
- Classification modeling
- Class imbalance handling
- Model comparison
- Hyperparameter tuning
- ROC-AUC, F1-score, precision, and recall evaluation
- Threshold analysis
- Business interpretation
- Technical documentation

[Student Dropout Prediction Technical Report](docs/Student_Dropout_Prediction_Technical_Report.pdf)
