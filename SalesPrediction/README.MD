# Task: Sales Prediction Using Python

## Project Overview

This project develops a machine learning system to predict product sales based on advertising expenditure across three marketing channels:

- TV advertising
- Radio advertising
- Newspaper advertising

The system performs exploratory data analysis, trains and evaluates multiple regression models, selects the best-performing model using cross-validation RMSE, saves the trained model, and provides an interactive interface for making sales predictions.

## Dataset

The project uses the **Advertising dataset** from *An Introduction to Statistical Learning (ISLR)*.

- **Dataset:** [Advertising.csv](https://www.statlearning.com/s/Advertising.csv)
- **Official resource:** [An Introduction to Statistical Learning – Data](https://trevorhastie.github.io/ISLR/data.html)
- **Records:** 200
- **Features:** TV, radio, newspaper
- **Target:** Sales

Advertising budgets are measured in thousands of US dollars. Sales are measured in thousands of units.

The dataset contains 200 valid records, with no missing values or duplicate rows detected during validation.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Project Structure

```text
SalesPrediction/
│
├── dataset/
│   └── Advertising.csv
│
├── models/
│   └── best_sales_model.joblib
│
├── src/
│   └── sales_prediction.py
│
├── reports/
│   ├── figures/
│   └── metrics/
│
└── README.MD
```

The `reports/figures/` and `reports/metrics/` directories are generated when the analysis and model evaluation are executed.

## Exploratory Data Analysis

The project examines the distribution of advertising expenditure and sales, feature relationships, and correlations.

The observed correlations with sales were:

| Advertising Channel | Correlation with Sales |
|---|---:|
| TV | 0.7822 |
| Radio | 0.5762 |
| Newspaper | 0.2283 |

TV advertising had the strongest linear association with sales in this dataset, followed by radio and newspaper advertising. These correlations describe relationships in the observed data and do not establish causation.

Visualizations include:
- Feature and target distributions
- Advertising expenditure versus sales
- Feature correlation heatmap
- Model evaluation plots

## Machine Learning Models

Three regression algorithms were trained and compared:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

The dataset was divided into training and testing subsets using an 80:20 split. Five-fold cross-validation was performed on the training subset. The model with the lowest mean cross-validation RMSE was selected.

## Model Evaluation

### Cross-Validation Results

| Model | CV RMSE | CV MAE | CV R² |
|---|---:|---:|---:|
| Linear Regression | 1.6758 | 1.2371 | 0.8703 |
| Random Forest | **0.8950** | **0.6413** | **0.9660** |
| Gradient Boosting | 0.9067 | 0.6535 | 0.9641 |

### Test Set Results

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Linear Regression | 1.4608 | 1.7816 | 0.8994 |
| Random Forest | **0.6139** | **0.7406** | **0.9826** |
| Gradient Boosting | 0.6486 | 0.7527 | 0.9821 |

### Selected Model

**Random Forest Regressor** was selected because it achieved the lowest mean cross-validation RMSE.

Its test-set performance was:

- MAE: 0.6139 thousand units
- RMSE: 0.7406 thousand units
- R²: 0.9826

The results indicate strong predictive performance on the held-out test subset. However, the dataset is relatively small, containing only 200 observations, so performance should be validated on additional data before relying on the model in a real business environment.

## Model Persistence

The selected model is saved using Joblib:

```text
SalesPrediction/models/best_sales_model.joblib
```

The saved artifact includes the trained model and associated metadata required by the prediction functionality.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/vnyabuto/OIBSIP.git
cd OIBSIP
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Running the Application

Run the main Python script from the project root:

```bash
python SalesPrediction/src/sales_prediction.py
```

Alternatively, from the `SalesPrediction` directory:

```bash
python src/sales_prediction.py
```

The application provides three menu options:

1. **Train and evaluate models** – performs exploratory analysis, trains the regression models, compares their performance, and saves the selected model.
2. **Predict sales using saved model** – loads the saved model and accepts advertising budgets interactively.
3. **Exit** – closes the application.

### Interactive Prediction Example

Example input:

```text
TV advertising budget: 200
Radio advertising budget: 30
Newspaper advertising budget: 40
```

Example output from the trained model:

```text
Estimated sales: 18.30 thousand units
Estimated units: 18,297
```

The advertising budget inputs are in thousands of US dollars, and the prediction is returned in thousands of units and converted to individual units for readability.

## Limitations

- The dataset contains only 200 observations.
- The data represents a historical advertising and sales relationship and may not reflect current market conditions.
- The model estimates associations from historical data; it does not establish that increasing advertising expenditure will cause a particular sales increase.
- Performance on unseen data may differ from the reported test results.
- The model should be retrained and validated when new, representative data becomes available.

## Author

Developed as part of the **Oasis Infobyte Data Science Internship (OIBSIP)**.

GitHub: [vnyabuto](https://github.com/vnyabuto)