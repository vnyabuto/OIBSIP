
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "dataset" / "Advertising.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
METRIC_DIR = REPORT_DIR / "metrics"

FEATURES = ["tv", "radio", "newspaper"]
TARGET = "sales"

RANDOM_STATE = 42

for directory in [MODEL_DIR, FIGURE_DIR, METRIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    data = pd.read_csv(DATASET_PATH)

    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
    )

    data = data.loc[
        :, ~data.columns.str.startswith("unnamed")
    ]

    missing_columns = [
        column for column in FEATURES + [TARGET]
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data[FEATURES + [TARGET]].copy()

    for column in FEATURES + [TARGET]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    initial_rows = len(data)

    data = data.dropna()
    data = data.drop_duplicates()

    if data.empty:
        raise ValueError("No valid records remain.")

    if (data[FEATURES + [TARGET]] < 0).any().any():
        raise ValueError(
            "Negative values found in the dataset."
        )

    print("\nDATASET INFORMATION")
    print("=" * 55)
    print(f"Original records: {initial_rows}")
    print(f"Valid records: {len(data)}")
    print(f"Removed records: {initial_rows - len(data)}")
    print(f"Features: {FEATURES}")
    print(f"Target: {TARGET}")

    print("\nData types:")
    print(data.dtypes)

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nDescriptive statistics:")
    print(data.describe().round(3))

    return data


def exploratory_analysis(data):
    print("\nEXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    print("\nCorrelation matrix:")
    print(data.corr().round(3))

    correlation = data.corr()[TARGET].drop(TARGET)

    print("\nCorrelation with sales:")
    print(correlation.sort_values(ascending=False))

    plt.figure(figsize=(9, 6))
    sns.heatmap(
        data.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Advertising Dataset Correlation Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "correlation_matrix.png",
        dpi=300
    )
    plt.close()

    for column in FEATURES + [TARGET]:
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data[column],
            kde=True,
            bins=20
        )
        plt.title(f"Distribution of {column.title()}")
        plt.xlabel(column.title())
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / f"{column}_distribution.png",
            dpi=300
        )
        plt.close()

    for feature in FEATURES:
        plt.figure(figsize=(8, 5))
        sns.regplot(
            data=data,
            x=feature,
            y=TARGET,
            scatter_kws={"alpha": 0.65},
            line_kws={"color": "red"}
        )
        plt.title(
            f"{feature.title()} Advertising vs Sales"
        )
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / f"{feature}_vs_sales.png",
            dpi=300
        )
        plt.close()

    print(
        f"\nEDA visualizations saved to: {FIGURE_DIR}"
    )


def build_models():
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            min_samples_split=2,
            loss="squared_error",
            random_state=RANDOM_STATE
        )
    }

    return models


def train_and_evaluate(data):
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    models = build_models()

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }

    results = []
    fitted_models = {}

    print("\nMODEL TRAINING AND CROSS-VALIDATION")
    print("=" * 65)

    for name, model in models.items():
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        test_mae = mean_absolute_error(
            y_test, predictions
        )

        test_rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        test_r2 = r2_score(
            y_test, predictions
        )

        cv_rmse = -cv_results["test_rmse"].mean()
        cv_mae = -cv_results["test_mae"].mean()
        cv_r2 = cv_results["test_r2"].mean()

        results.append({
            "Model": name,
            "CV_RMSE": cv_rmse,
            "CV_RMSE_Std": cv_results["test_rmse"].std(),
            "CV_MAE": cv_mae,
            "CV_R2": cv_r2,
            "Test_MAE": test_mae,
            "Test_RMSE": test_rmse,
            "Test_R2": test_r2
        })

        fitted_models[name] = model

        print(f"\n{name}")
        print(f"Cross-validation RMSE: {cv_rmse:.4f}")
        print(f"Cross-validation MAE: {cv_mae:.4f}")
        print(f"Cross-validation R2: {cv_r2:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test R2: {test_r2:.4f}")

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="CV_RMSE",
        ascending=True
    )

    comparison.to_csv(
        METRIC_DIR / "model_comparison.csv",
        index=False
    )

    best_name = comparison.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    print("\nMODEL COMPARISON")
    print("=" * 65)
    print(comparison.round(4).to_string(index=False))

    print(f"\nSelected model: {best_name}")

    predictions = best_model.predict(X_test)

    prediction_results = pd.DataFrame({
        "Actual_Sales": y_test.values,
        "Predicted_Sales": predictions,
        "Residual": y_test.values - predictions
    })

    prediction_results.to_csv(
        METRIC_DIR / "test_predictions.csv",
        index=False
    )

    best_result = comparison.iloc[0]

    metrics = {
        "selected_model": best_name,
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "cross_validation_rmse": float(
            best_result["CV_RMSE"]
        ),
        "test_mae": float(best_result["Test_MAE"]),
        "test_rmse": float(best_result["Test_RMSE"]),
        "test_r2": float(best_result["Test_R2"]),
        "features": FEATURES,
        "target": TARGET
    }

    with open(
        METRIC_DIR / "training_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    model_artifact = {
        "model": best_model,
        "model_name": best_name,
        "features": FEATURES,
        "target": TARGET,
        "target_unit": "thousands of units",
        "budget_unit": "thousands of USD"
    }

    joblib.dump(
        model_artifact,
        MODEL_DIR / "best_sales_model.joblib"
    )

    create_evaluation_plots(
        comparison,
        y_test,
        predictions,
        prediction_results
    )

    print("\nModel saved successfully.")
    print(f"Model location: {MODEL_DIR}")

    return best_model, comparison


def create_evaluation_plots(
    comparison,
    y_test,
    predictions,
    prediction_results
):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=comparison,
        x="Model",
        y="CV_RMSE"
    )
    plt.title("Model Comparison - Cross-Validation RMSE")
    plt.ylabel("Mean RMSE")
    plt.xlabel("Regression Model")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "model_comparison.png",
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    lower = min(y_test.min(), predictions.min())
    upper = max(y_test.max(), predictions.max())

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--"
    )

    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "actual_vs_predicted.png",
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(
        prediction_results["Residual"],
        kde=True,
        bins=15
    )
    plt.axvline(0, linestyle="--")
    plt.title("Prediction Residual Distribution")
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "residual_distribution.png",
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        predictions,
        prediction_results["Residual"],
        alpha=0.7
    )
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Sales")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted Sales")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "residuals_vs_predictions.png",
        dpi=300
    )
    plt.close()


def predict_sales():
    model_path = MODEL_DIR / "best_sales_model.joblib"

    if not model_path.exists():
        print("\nNo trained model found. Train a model first.")
        return

    artifact = joblib.load(model_path)
    model = artifact["model"]

    print("\nINTERACTIVE SALES PREDICTION")
    print("=" * 45)
    print("Enter advertising budgets in thousands of USD.")
    print("Enter 0 for TV budget to stop.")

    while True:
        try:
            tv = float(input("\nTV advertising budget: "))

            if tv == 0:
                print("Prediction session ended.")
                break

            radio = float(
                input("Radio advertising budget: ")
            )

            newspaper = float(
                input("Newspaper advertising budget: ")
            )

            budgets = [tv, radio, newspaper]

            if any(value < 0 for value in budgets):
                print("Budgets cannot be negative.")
                continue

            input_data = pd.DataFrame(
                [budgets],
                columns=FEATURES
            )

            prediction = model.predict(input_data)[0]

            print("\nPREDICTION RESULT")
            print("-" * 35)
            print(
                f"Estimated sales: {prediction:.2f} "
                "thousand units"
            )
            print(
                f"Estimated units: {prediction * 1000:,.0f}"
            )

        except ValueError:
            print("Invalid input. Please enter numeric values.")

        except KeyboardInterrupt:
            print("\nPrediction session ended.")
            break


def main():
    print("\nSALES PREDICTION USING MACHINE LEARNING")
    print("=" * 55)

    while True:
        print("\n1. Train and evaluate models")
        print("2. Predict sales using saved model")
        print("3. Exit")

        choice = input("\nSelect an option: ").strip()

        if choice == "1":
            try:
                data = load_dataset()
                exploratory_analysis(data)
                train_and_evaluate(data)
            except Exception as error:
                print(f"\nError: {error}")

        elif choice == "2":
            predict_sales()

        elif choice == "3":
            print("Application closed.")
            break

        else:
            print("Invalid selection. Choose 1, 2 or 3.")


if __name__ == "__main__":
    main()
