# Oasis Infobyte Data Science Internship (OIBSIP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Field-Data%20Science-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![GitHub](https://img.shields.io/badge/GitHub-vnyabuto-black?logo=github)

## Overview

This repository contains projects completed as part of the **Oasis Infobyte Data Science Internship (OIBSIP)**.

The internship provided an opportunity to apply data science and machine learning concepts to practical problems involving data acquisition, preprocessing, exploratory data analysis, model development, evaluation, and prediction.

Three projects were selected and implemented using Python and its data science ecosystem.

## Completed Projects

### Task 3: Car Price Prediction

**Objective:** Develop a machine learning model to estimate the selling price of used cars based on vehicle characteristics.

**Key components:**
- Data cleaning and preprocessing.
- Exploratory data analysis.
- Feature engineering.
- Regression model development and comparison.
- Model evaluation using MAE, RMSE, and R².
- Currency conversion for INR, KES, and USD.

**Models implemented:**
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

**Project documentation:** [Car Price Prediction](CarPricePrediction/README.MD)

**Project directory:** [CarPricePrediction](CarPricePrediction/)

### Task 4: Email Spam Detection

**Objective:** Develop a machine learning classification system capable of distinguishing spam emails from legitimate emails.

**Key components:**
- Email dataset preparation and validation.
- Text preprocessing and feature extraction.
- TF-IDF vectorization.
- Classification model training and comparison.
- Confusion matrix and classification metric analysis.
- Error analysis, including false positives and false negatives.
- Interactive spam prediction.

**Models implemented:**
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Classification (Linear SVC)

**Dataset:** Apache SpamAssassin public email corpus.

**Project documentation:** [Email Spam Detection](EmailSpamDetection/README.MD)

**Project directory:** [EmailSpamDetection](EmailSpamDetection/)

### Task 5: Sales Prediction Using Python

**Objective:** Build a regression-based machine learning system to predict sales using advertising expenditure across TV, radio, and newspaper channels.

**Key components:**
- Dataset acquisition and validation.
- Exploratory data analysis.
- Correlation analysis and visualization.
- Regression model training and comparison.
- Cross-validation and test-set evaluation.
- Model persistence using Joblib.
- Interactive sales prediction.

**Models implemented:**
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

**Dataset:** Advertising dataset from *An Introduction to Statistical Learning (ISLR)*.

**Best-performing model:** Random Forest Regressor, selected using cross-validation RMSE.

**Project documentation:** [Sales Prediction](SalesPrediction/README.MD)

**Project directory:** [SalesPrediction](SalesPrediction/)

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Natural Language Processing | TF-IDF, Scikit-learn |
| Model Persistence | Joblib |
| Development Environment | PyCharm |
| Version Control | Git and GitHub |

## Repository Structure

```text
OIBSIP/
│
├── CarPricePrediction/
│   ├── dataset/
│   ├── models/
│   ├── src/
│   │   └── car_price_prediction.py
│   └── README.MD
│
├── EmailSpamDetection/
│   ├── dataset/
│   ├── models/
│   ├── src/
│   │   ├── error_analysis.py
│   │   ├── exploratory_analysis.py
│   │   ├── inspect_false_positives.py
│   │   ├── predict_spam.py
│   │   ├── prepare_dataset.py
│   │   └── train_spam_classifier.py
│   └── README.MD
│
├── SalesPrediction/
│   ├── dataset/
│   ├── models/
│   ├── src/
│   │   └── sales_prediction.py
│   ├── reports/
│   └── README.MD
│
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: Generated reports, model artifacts, and dataset files may vary depending on the project and Git tracking configuration. Raw and processed email data are excluded from version control.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vnyabuto/OIBSIP.git
cd OIBSIP
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

Refer to each project's README for its specific execution instructions and dataset requirements.

## Learning Outcomes

Through these projects, I gained practical experience in:

- Acquiring, cleaning, and validating datasets.
- Performing exploratory data analysis and visualization.
- Applying feature engineering and preprocessing techniques.
- Building and evaluating regression and classification models.
- Using cross-validation for model selection.
- Analyzing model performance and prediction errors.
- Saving and loading trained machine learning models.
- Structuring Python projects for reproducibility.
- Using Git and GitHub for project version control and documentation.

## Author

**Victor Nyabuto**

- GitHub: [@vnyabuto](https://github.com/vnyabuto)
- Repository: [OIBSIP](https://github.com/vnyabuto/OIBSIP)

## Acknowledgement

I would like to thank **Oasis Infobyte** for providing the opportunity to participate in the Data Science Internship and apply machine learning concepts through practical projects.

---

*Developed as part of the Oasis Infobyte Data Science Internship (OIBSIP).* 