
from pathlib import Path
from email import policy
from email.parser import BytesParser
from email.header import decode_header
from html.parser import HTMLParser
import re

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "dataset" / "raw"
OUTPUT_DIR = PROJECT_DIR / "dataset" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "emails.csv"

HAM_DIRECTORIES = ["easy_ham", "hard_ham"]
SPAM_DIRECTORY = "spam"

MESSAGE_FILENAME_PATTERN = re.compile(r"^\d+\.[0-9a-fA-F]+$")

VALID_EMAIL_HEADERS = {
    "from",
    "to",
    "cc",
    "bcc",
    "subject",
    "date",
    "message-id",
    "received",
    "return-path",
}


class EmailHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text_parts = []
        self.ignored_tags = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style"}:
            self.ignored_tags.append(tag.lower())

    def handle_endtag(self, tag):
        if self.ignored_tags and tag.lower() == self.ignored_tags[-1]:
            self.ignored_tags.pop()

    def handle_data(self, data):
        if not self.ignored_tags:
            cleaned = data.strip()
            if cleaned:
                self.text_parts.append(cleaned)

    def get_text(self):
        return " ".join(self.text_parts)


def decode_bytes(payload, charset=None):
    if not payload:
        return ""

    encodings = []

    if charset:
        encodings.append(charset)

    encodings.extend(["utf-8", "latin-1"])

    for encoding in encodings:
        try:
            return payload.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    return payload.decode("utf-8", errors="replace")


def decode_mime_header(value):
    if not value:
        return ""

    decoded_parts = []

    try:
        parts = decode_header(value)
    except Exception:
        return str(value)

    for part, charset in parts:
        if isinstance(part, bytes):
            decoded_parts.append(decode_bytes(part, charset))
        else:
            decoded_parts.append(str(part))

    return " ".join(decoded_parts).strip()


def extract_part_text(part):
    payload = part.get_payload(decode=True)

    if payload is None:
        raw_payload = part.get_payload()

        if isinstance(raw_payload, str):
            return raw_payload.strip()

        return ""

    return decode_bytes(payload, part.get_content_charset()).strip()


def html_to_text(html_content):
    parser = EmailHTMLParser()

    try:
        parser.feed(html_content)
        parser.close()
        return parser.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html_content).strip()


def extract_email_body(message):
    plain_text_parts = []
    html_parts = []

    if message.is_multipart():
        parts = message.walk()
    else:
        parts = [message]

    for part in parts:
        if part.is_multipart():
            continue

        if part.get_content_disposition() == "attachment":
            continue

        content_type = part.get_content_type().lower()

        if content_type not in {"text/plain", "text/html"}:
            continue

        content = extract_part_text(part)

        if not content:
            continue

        if content_type == "text/plain":
            plain_text_parts.append(content)
        elif content_type == "text/html":
            html_parts.append(content)

    if plain_text_parts:
        return "\n".join(plain_text_parts).strip()

    if html_parts:
        converted_parts = [html_to_text(part) for part in html_parts]
        return "\n".join(converted_parts).strip()

    return ""


def has_email_headers(message):
    existing_headers = {
        header.lower()
        for header in message.keys()
    }

    return bool(existing_headers.intersection(VALID_EMAIL_HEADERS))


def is_valid_message_file(file_path):
    return bool(MESSAGE_FILENAME_PATTERN.fullmatch(file_path.name))


def parse_email(file_path, label):
    try:
        with file_path.open("rb") as email_file:
            message = BytesParser(policy=policy.default).parse(email_file)

        if not has_email_headers(message):
            return None

        subject = decode_mime_header(message.get("Subject", ""))
        body = extract_email_body(message)

        text = " ".join(
            value for value in [subject, body] if value
        ).strip()

        if not text:
            return None

        word_count = len(text.split())
        character_count = len(text)
        subject_word_count = len(subject.split())
        body_word_count = len(body.split())

        return {
            "subject": subject,
            "body": body,
            "text": text,
            "label": label,
            "source_file": str(file_path.relative_to(RAW_DIR)),
            "word_count": word_count,
            "character_count": character_count,
            "subject_word_count": subject_word_count,
            "body_word_count": body_word_count,
        }

    except Exception as error:
        print(f"Could not process {file_path.name}: {error}")
        return None


def collect_emails():
    records = []
    invalid_filenames = 0
    invalid_messages = 0

    for directory_name in HAM_DIRECTORIES:
        directory = RAW_DIR / directory_name

        if not directory.exists():
            raise FileNotFoundError(
                f"Required dataset directory not found: {directory}"
            )

        for file_path in sorted(directory.iterdir()):
            if not file_path.is_file():
                continue

            if not is_valid_message_file(file_path):
                invalid_filenames += 1
                continue

            record = parse_email(file_path, "ham")

            if record is None:
                invalid_messages += 1
                continue

            records.append(record)

    spam_directory = RAW_DIR / SPAM_DIRECTORY

    if not spam_directory.exists():
        raise FileNotFoundError(
            f"Required dataset directory not found: {spam_directory}"
        )

    for file_path in sorted(spam_directory.iterdir()):
        if not file_path.is_file():
            continue

        if not is_valid_message_file(file_path):
            invalid_filenames += 1
            continue

        record = parse_email(file_path, "spam")

        if record is None:
            invalid_messages += 1
            continue

        records.append(record)

    return records, invalid_filenames, invalid_messages


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {RAW_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SPAMASSASSIN EMAIL DATASET PREPARATION")
    print("=" * 70)

    records, invalid_filenames, invalid_messages = collect_emails()

    if not records:
        raise ValueError(
            "No valid emails were extracted. Check the raw dataset directories."
        )

    dataframe = pd.DataFrame(records)

    expected_columns = [
        "subject",
        "body",
        "text",
        "label",
        "source_file",
        "word_count",
        "character_count",
        "subject_word_count",
        "body_word_count",
    ]

    dataframe = dataframe[expected_columns]
    dataframe.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    class_counts = dataframe["label"].value_counts()
    total_emails = len(dataframe)

    print(f"\nTotal valid emails processed: {total_emails}")
    print(f"Ham emails: {class_counts.get('ham', 0)}")
    print(f"Spam emails: {class_counts.get('spam', 0)}")

    print("\nClass distribution:")
    for label in ["ham", "spam"]:
        count = class_counts.get(label, 0)
        percentage = count / total_emails * 100
        print(f"{label.capitalize()}: {count} ({percentage:.2f}%)")

    print("\nDataset quality checks:")
    print(f"Files rejected by filename validation: {invalid_filenames}")
    print(f"Files rejected by email validation: {invalid_messages}")

    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"Output columns: {len(dataframe.columns)}")

    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    main()
