
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_FILE = PROJECT_DIR / "dataset" / "processed" / "emails.csv"
PREDICTIONS_FILE = (
    PROJECT_DIR / "reports" / "metrics" / "test_predictions.csv"
)

OUTPUT_DIR = PROJECT_DIR / "reports" / "error_analysis"

RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_COLUMN = "linear_svm_prediction"


def load_and_reconstruct_test_data():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Test predictions not found: {PREDICTIONS_FILE}"
        )

    dataframe = pd.read_csv(DATASET_FILE)

    required_columns = {
        "text",
        "label",
        "subject",
        "body",
        "source_file"
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    dataframe = dataframe.dropna(subset=["text", "label"]).copy()

    dataframe["text"] = dataframe["text"].astype(str)
    dataframe["label"] = (
        dataframe["label"].astype(str).str.lower().str.strip()
    )

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

    dataframe = dataframe.drop_duplicates(
        subset=["normalized_text"],
        keep="first"
    ).copy()

    dataframe["label_encoded"] = dataframe["label"].map({
        "ham": 0,
        "spam": 1
    })

    dataframe = dataframe.reset_index(drop=True)

    X = dataframe["text"]
    y = dataframe["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    test_dataframe = dataframe.loc[X_test.index].copy()

    test_dataframe = test_dataframe.reset_index(drop=True)

    predictions = pd.read_csv(PREDICTIONS_FILE)

    if MODEL_COLUMN not in predictions.columns:
        raise ValueError(
            f"Prediction file does not contain {MODEL_COLUMN}."
        )

    if len(test_dataframe) != len(predictions):
        raise ValueError(
            "The reconstructed test dataset and prediction file "
            "have different numbers of records."
        )

    test_dataframe["actual_label"] = y_test.map({
        0: "ham",
        1: "spam"
    }).values

    test_dataframe["predicted_label"] = (
        predictions[MODEL_COLUMN].astype(str).str.lower().str.strip()
    )

    if not (
        test_dataframe["actual_label"].values
        == predictions["actual_label"].astype(str).str.lower().str.strip().values
    ).all():
        raise ValueError(
            "Actual labels do not match. The dataset or training split "
            "may have changed since model training."
        )

    valid_labels = {"ham", "spam"}

    if not set(test_dataframe["predicted_label"]).issubset(valid_labels):
        raise ValueError("Unexpected prediction labels detected.")

    return test_dataframe


def analyze_errors(test_dataframe):
    test_dataframe["result"] = "Correct"

    test_dataframe.loc[
        (test_dataframe["actual_label"] == "ham")
        & (test_dataframe["predicted_label"] == "spam"),
        "result"
    ] = "False Positive"

    test_dataframe.loc[
        (test_dataframe["actual_label"] == "spam")
        & (test_dataframe["predicted_label"] == "ham"),
        "result"
    ] = "False Negative"

    false_positives = test_dataframe[
        test_dataframe["result"] == "False Positive"
    ].copy()

    false_negatives = test_dataframe[
        test_dataframe["result"] == "False Negative"
    ].copy()

    return false_positives, false_negatives


def save_error_reports(
    test_dataframe,
    false_positives,
    false_negatives
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "source_file",
        "subject",
        "actual_label",
        "predicted_label",
        "result",
        "text"
    ]

    false_positives[columns].to_csv(
        OUTPUT_DIR / "false_positives.csv",
        index=False
    )

    false_negatives[columns].to_csv(
        OUTPUT_DIR / "false_negatives.csv",
        index=False
    )

    all_errors = test_dataframe[
        test_dataframe["result"] != "Correct"
    ]

    all_errors[columns].to_csv(
        OUTPUT_DIR / "all_misclassified_emails.csv",
        index=False
    )

    matrix = confusion_matrix(
        test_dataframe["actual_label"],
        test_dataframe["predicted_label"],
        labels=["ham", "spam"]
    )

    tn, fp, fn, tp = matrix.ravel()

    total = len(test_dataframe)

    summary = {
        "total_test_emails": int(total),
        "correct_predictions": int(tn + tp),
        "incorrect_predictions": int(fp + fn),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": float(fp / (fp + tn)),
        "false_negative_rate": float(fn / (fn + tp)),
        "test_accuracy": float((tn + tp) / total)
    }

    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "error_analysis_summary.csv",
        index=False
    )

    return summary


def display_examples(dataframe, title, limit=5):
    print(f"\n{title}")
    print("=" * 70)

    if dataframe.empty:
        print("No examples found.")
        return

    for index, (_, row) in enumerate(dataframe.head(limit).iterrows(), 1):
        print(f"\nExample {index}")
        print(f"Source: {row['source_file']}")
        print(f"Subject: {row['subject']}")
        print(f"Actual: {row['actual_label']}")
        print(f"Predicted: {row['predicted_label']}")
        print(f"Email text: {str(row['text'])[:700]}")
        print("-" * 70)


def main():
    print("=" * 70)
    print("EMAIL SPAM DETECTION - ERROR ANALYSIS")
    print("=" * 70)

    test_dataframe = load_and_reconstruct_test_data()

    false_positives, false_negatives = analyze_errors(
        test_dataframe
    )

    summary = save_error_reports(
        test_dataframe,
        false_positives,
        false_negatives
    )

    print("\nERROR ANALYSIS SUMMARY")
    print("=" * 70)

    for metric, value in summary.items():
        if isinstance(value, float):
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"{metric.replace('_', ' ').title()}: {value}")

    print("\nCLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            test_dataframe["actual_label"],
            test_dataframe["predicted_label"],
            labels=["ham", "spam"],
            zero_division=0
        )
    )

    display_examples(
        false_positives,
        "FALSE POSITIVES - HAM CLASSIFIED AS SPAM"
    )

    display_examples(
        false_negatives,
        "FALSE NEGATIVES - SPAM CLASSIFIED AS HAM"
    )

    print("\nERROR REPORTS SAVED")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print("False positives: false_positives.csv")
    print("False negatives: false_negatives.csv")
    print("All errors: all_misclassified_emails.csv")
    print("Summary: error_analysis_summary.csv")

    print("\nERROR ANALYSIS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
