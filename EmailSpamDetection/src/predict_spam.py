
from pathlib import Path

import joblib

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_DIR / "models" / "best_spam_classifier.joblib"


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    return joblib.load(MODEL_FILE)


def classify_email(model, subject, body):
    email_text = f"{subject} {body}".strip()

    if not email_text:
        print("\nPlease provide an email subject or body.")
        return

    prediction = model.predict([email_text])[0]
    decision_score = model.decision_function([email_text])[0]

    print("\n" + "=" * 60)
    print("EMAIL SPAM DETECTION RESULT")
    print("=" * 60)

    if prediction == 1:
        print("Classification: SPAM")
        print("Status: This email is likely unwanted or unsolicited.")
    else:
        print("Classification: HAM")
        print("Status: This email is likely legitimate.")

    print(f"Model decision score: {decision_score:.4f}")

    if decision_score > 0:
        print("Decision direction: Spam")
    else:
        print("Decision direction: Ham")

    print("\nNote: The decision score is not a probability.")
    print("Always review suspicious emails before interacting with them.")
    print("=" * 60)


def main():
    print("=" * 60)
    print("EMAIL SPAM DETECTION SYSTEM")
    print("Powered by Linear Support Vector Machine")
    print("=" * 60)

    try:
        model = load_model()
    except Exception as error:
        print(f"Model loading failed: {error}")
        return

    print("\nModel loaded successfully.")
    print("\nEnter an email to classify.")
    print("Type 'exit' as the subject to close the application.")

    while True:
        print("\n" + "-" * 60)

        subject = input("Email subject: ").strip()

        if subject.lower() == "exit":
            print("\nExiting Email Spam Detection System.")
            break

        body = input("Email body: ").strip()

        if not subject and not body:
            print("Please enter an email subject or body.")
            continue

        classify_email(model, subject, body)


if __name__ == "__main__":
    main()
