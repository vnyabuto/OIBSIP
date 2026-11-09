
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "reports"
    / "error_analysis"
    / "false_positives.csv"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    dataframe = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print("FALSE POSITIVE INVESTIGATION")
    print("=" * 70)

    print(f"\nTotal false positives: {len(dataframe)}")

    for index, row in dataframe.iterrows():
        print(f"\nEMAIL {index + 1}")
        print("-" * 70)
        print(f"Source: {row['source_file']}")
        print(f"Subject: {row['subject']}")
        print(f"Actual: {row['actual_label']}")
        print(f"Predicted: {row['predicted_label']}")
        print(f"Text preview: {str(row['text'])[:500]}")

    print("\nINSPECTION COMPLETED")


if __name__ == "__main__":
    main()
