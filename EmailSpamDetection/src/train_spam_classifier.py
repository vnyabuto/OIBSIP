
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score
)
from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
    train_test_split
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_DIR / "dataset" / "processed" / "emails.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
METRICS_DIR = REPORT_DIR / "metrics"
FIGURES_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TFIDF_CONFIG = {
    "stop_words": "english",
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.98,
    "strip_accents": "unicode",
    "sublinear_tf": True
}


def create_directories():
    for directory in [MODEL_DIR, METRICS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    dataframe = pd.read_csv(DATASET_FILE)

    required_columns = {"text", "label"}

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            f"Dataset must contain these columns: {required_columns}"
        )

    dataframe = dataframe.dropna(subset=["text", "label"]).copy()
    dataframe["text"] = dataframe["text"].astype(str)
    dataframe["label"] = dataframe["label"].astype(str).str.lower().str.strip()

    dataframe = dataframe[
        dataframe["label"].isin(["ham", "spam"])
    ].copy()

    dataframe["normalized_text"] = (
        dataframe["text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )

    dataframe = dataframe[
        dataframe["normalized_text"].str.len() > 0
    ].copy()

    original_count = len(dataframe)

    dataframe = dataframe.drop_duplicates(
        subset=["normalized_text"],
        keep="first"
    ).copy()

    duplicates_removed = original_count - len(dataframe)

    dataframe["label_encoded"] = dataframe["label"].map({
        "ham": 0,
        "spam": 1
    })

    dataframe = dataframe.reset_index(drop=True)

    if dataframe["label_encoded"].nunique() != 2:
        raise ValueError("Both ham and spam classes must be present.")

    return dataframe, duplicates_removed


def build_models():
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }

    pipelines = {}

    for name, classifier in models.items():
        pipelines[name] = Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
            ("classifier", classifier)
        ])

    return pipelines


def evaluate_models(X_train, X_test, y_train, y_test, pipelines):
    scoring = {
        "f1": make_scorer(f1_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "accuracy": "accuracy"
    }

    cross_validator = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []
    fitted_models = {}
    predictions = {}
    prediction_scores = {}
    classification_reports = {}

    print("\nMODEL TRAINING AND EVALUATION")
    print("=" * 70)

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise"
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        if hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(X_test)[:, 1]
        else:
            scores = pipeline.decision_function(X_test)

        fitted_models[name] = pipeline
        predictions[name] = y_pred
        prediction_scores[name] = scores

        report = classification_report(
            y_test,
            y_pred,
            target_names=["ham", "spam"],
            output_dict=True,
            zero_division=0
        )

        classification_reports[name] = report

        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        test_recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        test_f1 = f1_score(
            y_test, y_pred, zero_division=0
        )
        average_precision = average_precision_score(
            y_test, scores
        )

        results.append({
            "Model": name,
            "CV F1 Mean": cv_results["test_f1"].mean(),
            "CV F1 Std": cv_results["test_f1"].std(),
            "CV Precision Mean": cv_results["test_precision"].mean(),
            "CV Recall Mean": cv_results["test_recall"].mean(),
            "CV Accuracy Mean": cv_results["test_accuracy"].mean(),
            "Test Accuracy": test_accuracy,
            "Test Precision": test_precision,
            "Test Recall": test_recall,
            "Test F1": test_f1,
            "Test Average Precision": average_precision
        })

        print(f"Cross-validation F1: {cv_results['test_f1'].mean():.4f}")
        print(f"Test Accuracy:       {test_accuracy:.4f}")
        print(f"Test Precision:      {test_precision:.4f}")
        print(f"Test Recall:         {test_recall:.4f}")
        print(f"Test F1-score:       {test_f1:.4f}")
        print(f"Test Average Precision: {average_precision:.4f}")

    results_dataframe = pd.DataFrame(results)

    results_dataframe = results_dataframe.sort_values(
        by="CV F1 Mean",
        ascending=False
    ).reset_index(drop=True)

    best_model_name = results_dataframe.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    return (
        results_dataframe,
        fitted_models,
        predictions,
        prediction_scores,
        classification_reports,
        best_model_name,
        best_model
    )


def save_metrics(
    results_dataframe,
    predictions,
    classification_reports,
    y_test,
    best_model_name,
    best_model
):
    comparison_file = METRICS_DIR / "model_comparison.csv"
    predictions_file = METRICS_DIR / "test_predictions.csv"
    reports_file = METRICS_DIR / "classification_reports.json"

    results_dataframe.to_csv(comparison_file, index=False)

    prediction_dataframe = pd.DataFrame({
        "actual_label": y_test.map({0: "ham", 1: "spam"}).values
    })

    for name, y_pred in predictions.items():
        column_name = (
            name.lower()
            .replace(" ", "_")
        )

        prediction_dataframe[f"{column_name}_prediction"] = pd.Series(
            y_pred
        ).map({0: "ham", 1: "spam"}).values

    prediction_dataframe.to_csv(predictions_file, index=False)

    with reports_file.open("w", encoding="utf-8") as report_file:
        json.dump(
            classification_reports,
            report_file,
            indent=4
        )

    model_path = MODEL_DIR / "best_spam_classifier.joblib"
    joblib.dump(best_model, model_path)

    best_metrics = results_dataframe[
        results_dataframe["Model"] == best_model_name
    ].iloc[0].to_dict()

    summary = {
        "best_model": best_model_name,
        "selection_metric": "Mean cross-validation F1-score",
        "best_model_metrics": best_metrics,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cross_validation_folds": CV_FOLDS,
        "tfidf_configuration": {
            **TFIDF_CONFIG,
            "ngram_range": list(TFIDF_CONFIG["ngram_range"])
        },
        "generated_at": datetime.now().isoformat()
    }

    with (METRICS_DIR / "training_summary.json").open(
        "w",
        encoding="utf-8"
    ) as summary_file:
        json.dump(summary, summary_file, indent=4)

    print("\nSAVED ARTIFACTS")
    print("-" * 70)
    print(f"Model: {model_path}")
    print(f"Comparison: {comparison_file}")
    print(f"Predictions: {predictions_file}")
    print(f"Classification reports: {reports_file}")


def generate_visualizations(
    results_dataframe,
    predictions,
    prediction_scores,
    y_test
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for index, (_, row) in enumerate(results_dataframe.iterrows()):
        model_name = row["Model"]
        y_pred = predictions[model_name]

        matrix = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Ham", "Spam"]
        )

        display.plot(
            ax=axes[index],
            colorbar=False,
            values_format="d"
        )

        axes[index].set_title(model_name)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "confusion_matrices.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))

    for model_name, scores in prediction_scores.items():
        precision, recall, _ = precision_recall_curve(
            y_test,
            scores
        )

        average_precision = average_precision_score(
            y_test,
            scores
        )

        plt.plot(
            recall,
            precision,
            label=f"{model_name} (AP = {average_precision:.3f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "precision_recall_curves.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))

    plt.bar(
        results_dataframe["Model"],
        results_dataframe["CV F1 Mean"],
        yerr=results_dataframe["CV F1 Std"],
        capsize=5
    )

    plt.ylabel("Cross-validation F1-score")
    plt.xlabel("Classification Model")
    plt.title("Model Comparison - Cross-validation F1-score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()


def main():
    print("=" * 70)
    print("EMAIL SPAM DETECTION - MODEL TRAINING")
    print("=" * 70)

    create_directories()

    dataframe, duplicates_removed = load_dataset()

    X = dataframe["text"]
    y = dataframe["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nDATASET INFORMATION")
    print("-" * 70)
    print(f"Usable emails after duplicate removal: {len(dataframe)}")
    print(f"Duplicate email texts removed: {duplicates_removed}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Ham training records: {(y_train == 0).sum()}")
    print(f"Spam training records: {(y_train == 1).sum()}")
    print(f"Ham testing records: {(y_test == 0).sum()}")
    print(f"Spam testing records: {(y_test == 1).sum()}")

    pipelines = build_models()

    (
        results_dataframe,
        fitted_models,
        predictions,
        prediction_scores,
        classification_reports,
        best_model_name,
        best_model
    ) = evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        pipelines
    )

    print("\nMODEL COMPARISON")
    print("=" * 70)
    print(results_dataframe.to_string(index=False))

    print("\nBEST MODEL")
    print("=" * 70)
    print(f"Selected model: {best_model_name}")
    print(
        "Selection basis: Highest mean five-fold "
        "cross-validation F1-score."
    )

    save_metrics(
        results_dataframe,
        predictions,
        classification_reports,
        y_test,
        best_model_name,
        best_model
    )

    generate_visualizations(
        results_dataframe,
        predictions,
        prediction_scores,
        y_test
    )

    print("\nTRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Best model: {best_model_name}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Metrics directory: {METRICS_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
