
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_DIR / "dataset" / "processed" / "emails.csv"
REPORT_DIR = PROJECT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    df = pd.read_csv(DATASET_FILE)
    df["text"] = df["text"].fillna("")
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    return df


def prepare_features(df):
    df["word_count"] = df["text"].str.split().str.len()
    df["character_count"] = df["text"].str.len()
    df["subject_word_count"] = df["subject"].str.split().str.len()
    df["body_word_count"] = df["body"].str.split().str.len()

    return df


def dataset_overview(df):
    print("\nDATASET OVERVIEW")
    print("=" * 60)

    print(f"Total emails: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nClass distribution:")
    print(df["label"].value_counts())

    print("\nClass percentages:")
    print(
        (df["label"].value_counts(normalize=True) * 100).round(2)
    )

    print("\nMissing values:")
    print(df.isnull().sum())

    print(f"\nExact duplicate records: {df.duplicated().sum()}")
    print(
        "Duplicate email texts:",
        df.duplicated(subset=["text"]).sum()
    )

    print("\nMessage statistics:")
    print(
        df.groupby("label")[
            ["word_count", "character_count",
             "subject_word_count", "body_word_count"]
        ].agg(["mean", "median", "min", "max"]).round(2)
    )


def plot_class_distribution(df):
    plt.figure(figsize=(8, 5))

    ax = sns.countplot(
        data=df,
        x="label",
        hue="label",
        palette="Set2",
        legend=False
    )

    for container in ax.containers:
        ax.bar_label(container)

    plt.title("Email Class Distribution")
    plt.xlabel("Email Category")
    plt.ylabel("Number of Emails")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "class_distribution.png",
        dpi=300
    )
    plt.show()
    plt.close()


def plot_word_count_distribution(df):
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="word_count",
        hue="label",
        bins=50,
        kde=True,
        element="step",
        stat="density",
        common_norm=False
    )

    plt.title("Email Word Count Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Density")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "word_count_distribution.png",
        dpi=300
    )
    plt.show()
    plt.close()


def plot_character_distribution(df):
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="label",
        y="character_count",
        hue="label",
        palette="Set2",
        legend=False,
        showfliers=False
    )

    plt.title("Email Length by Class")
    plt.xlabel("Email Category")
    plt.ylabel("Number of Characters")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "email_length_comparison.png",
        dpi=300
    )
    plt.show()
    plt.close()


def plot_subject_length(df):
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="label",
        y="subject_word_count",
        hue="label",
        palette="Set2",
        legend=False,
        showfliers=False
    )

    plt.title("Email Subject Length by Class")
    plt.xlabel("Email Category")
    plt.ylabel("Subject Word Count")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "subject_length_comparison.png",
        dpi=300
    )
    plt.show()
    plt.close()


def plot_top_words(df, label, filename):
    class_text = df.loc[
        df["label"] == label, "text"
    ].astype(str)

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=20,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
    )

    word_matrix = vectorizer.fit_transform(class_text)

    word_counts = pd.DataFrame({
        "word": vectorizer.get_feature_names_out(),
        "frequency": word_matrix.sum(axis=0).A1
    })

    word_counts = word_counts.sort_values(
        "frequency",
        ascending=False
    )

    plt.figure(figsize=(11, 7))

    sns.barplot(
        data=word_counts,
        x="frequency",
        y="word",
        hue="word",
        palette="viridis",
        legend=False
    )

    plt.title(f"Top 20 Words in {label.upper()} Emails")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300
    )
    plt.show()
    plt.close()


def save_summary(df):
    summary = df.groupby("label").agg(
        email_count=("text", "count"),
        average_word_count=("word_count", "mean"),
        median_word_count=("word_count", "median"),
        average_character_count=("character_count", "mean"),
        average_subject_words=("subject_word_count", "mean")
    ).round(2)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        REPORT_DIR / "dataset_summary.csv"
    )


def main():
    print("=" * 60)
    print("EMAIL SPAM DETECTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = load_dataset()
    df = prepare_features(df)

    dataset_overview(df)

    print("\nGENERATING VISUALIZATIONS")
    print("-" * 60)

    plot_class_distribution(df)
    plot_word_count_distribution(df)
    plot_character_distribution(df)
    plot_subject_length(df)
    plot_top_words(df, "ham", "top_ham_words.png")
    plot_top_words(df, "spam", "top_spam_words.png")

    save_summary(df)

    print("\nEDA COMPLETED SUCCESSFULLY")
    print(f"Reports directory: {REPORT_DIR}")
    print(f"Visualizations directory: {FIGURE_DIR}")


if __name__ == "__main__":
    main()

