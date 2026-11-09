
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dataset" / "CarDekho_Dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

required_columns = [
    "Year",
    "Selling_Price",
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner"
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df.drop_duplicates().copy()

df["Car_Age"] = pd.Timestamp.now().year - df["Year"]

numerical_features = [
    "Car_Age",
    "Present_Price",
    "Kms_Driven",
    "Owner"
]

categorical_features = [
    "Fuel_Type",
    "Seller_Type",
    "Transmission"
]

target = "Selling_Price"

X = df[numerical_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

cv_strategy = RepeatedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)

results = []
trained_models = {}

print("\nDATA PREPARATION")
print("=" * 70)
print(f"Dataset records after duplicate removal: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Total input features: {X_train.shape[1]}")

print("\nMODEL TRAINING AND EVALUATION")
print("=" * 70)

for model_name, estimator in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator)
        ]
    )

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv_strategy,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    cv_rmse = -cv_scores.mean()
    cv_std = cv_scores.std()

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "CV_RMSE": cv_rmse,
        "CV_Std": cv_std,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[model_name] = pipeline

    print(f"\n{model_name}")
    print("-" * 70)
    print(f"Cross-validation RMSE: {cv_rmse:.4f}")
    print(f"Cross-validation Std: {cv_std:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV_RMSE",
    ascending=True
).reset_index(drop=True)

print("\nMODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

model_path = MODEL_DIR / "best_car_price_model.joblib"
results_path = MODEL_DIR / "model_comparison.csv"
predictions_path = MODEL_DIR / "test_predictions.csv"
residuals_path = MODEL_DIR / "residual_analysis.csv"

joblib.dump(best_model, model_path)

print("\nMODEL SELECTION")
print("=" * 70)
print(f"Best model based on repeated cross-validation: {best_model_name}")
print(f"Saved model: {model_path}")

best_predictions = best_model.predict(X_test)
residuals = y_test - best_predictions

print("\nRESIDUAL ANALYSIS")
print("=" * 70)
print(f"Mean residual: {residuals.mean():.4f}")
print(f"Residual standard deviation: {residuals.std():.4f}")
print(f"Maximum overprediction: {max(0, -residuals.min()):.4f}")
print(f"Maximum underprediction: {max(0, residuals.max()):.4f}")

prediction_results = pd.DataFrame({
    "Actual_Price": y_test,
    "Predicted_Price": best_predictions,
    "Residual": residuals
})

prediction_results.to_csv(predictions_path, index=False)

residual_results = pd.DataFrame({
    "Predicted_Price": best_predictions,
    "Residual": residuals
})

residual_results.to_csv(residuals_path, index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(
    x=y_test,
    y=best_predictions,
    ax=axes[0]
)

minimum = min(y_test.min(), best_predictions.min())
maximum = max(y_test.max(), best_predictions.max())

axes[0].plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

axes[0].set_title("Actual vs Predicted Prices")
axes[0].set_xlabel("Actual Selling Price")
axes[0].set_ylabel("Predicted Selling Price")

sns.scatterplot(
    x=best_predictions,
    y=residuals,
    ax=axes[1]
)

axes[1].axhline(y=0, linestyle="--")
axes[1].set_title("Residuals vs Predicted Prices")
axes[1].set_xlabel("Predicted Selling Price")
axes[1].set_ylabel("Residuals")

sns.histplot(
    residuals,
    kde=True,
    ax=axes[2]
)

axes[2].set_title("Residual Distribution")
axes[2].set_xlabel("Residual")

plt.tight_layout()
plt.show()

regressor = best_model.named_steps["regressor"]

if hasattr(regressor, "feature_importances_"):

    feature_names = best_model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance_values = regressor.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFEATURE IMPORTANCE")
    print("=" * 70)
    print(importance_df.to_string(index=False))

    importance_df.to_csv(
        MODEL_DIR / "feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(11, 7))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature"
    )

    plt.title(f"Feature Importance - {best_model_name}")
    plt.tight_layout()
    plt.show()

elif hasattr(regressor, "coef_"):

    feature_names = best_model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": regressor.coef_
    })

    importance_df["Absolute_Importance"] = (
        importance_df["Coefficient"].abs()
    )

    importance_df = importance_df.sort_values(
        by="Absolute_Importance",
        ascending=False
    )

    print("\nFEATURE COEFFICIENTS")
    print("=" * 70)
    print(importance_df.to_string(index=False))

    importance_df.to_csv(
        MODEL_DIR / "feature_importance.csv",
        index=False
    )

results_df.to_csv(results_path, index=False)

print("\nGENERATED FILES")
print("=" * 70)
print(f"Trained model: {model_path}")
print(f"Model comparison: {results_path}")
print(f"Test predictions: {predictions_path}")
print(f"Residual analysis: {residuals_path}")

if (MODEL_DIR / "feature_importance.csv").exists():
    print(f"Feature importance: {MODEL_DIR / 'feature_importance.csv'}")

print("\nCAR PRICE PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
