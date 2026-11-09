# Task: Email Spam Detection Using Machine Learning

## Project Overview

This project implements an email spam detection system using Natural Language Processing (NLP) and machine learning. It analyses email subject lines and message bodies, extracts textual features using TF-IDF, and classifies messages as either **Spam** or **Ham (Legitimate Email)**.

The project uses the Apache SpamAssassin Public Mail Corpus and evaluates three machine learning algorithms:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

Linear SVM was selected as the best-performing model based on its mean five-fold cross-validation F1-score.

## Project Objectives

- Obtain and prepare a real-world email dataset.
- Extract email subjects and message bodies.
- Perform exploratory data analysis.
- Apply text preprocessing and TF-IDF feature extraction.
- Train and compare multiple machine learning classifiers.
- Evaluate classification performance using appropriate metrics.
- Investigate false positives and false negatives.
- Develop an interactive command-line script for email classification.

## Dataset

**Source:** Apache SpamAssassin Public Mail Corpus

Official dataset website:

https://spamassassin.apache.org/old/publiccorpus/

The project uses three publicly available corpus archives:

| Dataset | Archive |
|---|---|
| Easy Ham | `20030228_easy_ham.tar.bz2` |
| Hard Ham | `20030228_hard_ham.tar.bz2` |
| Spam | `20030228_spam.tar.bz2` |

### Dataset Distribution

After email validation and dataset preparation:

| Category | Records |
|---|---:|
| Ham | 2,750 |
| Spam | 500 |
| **Total** | **3,250** |

The dataset is imbalanced, with legitimate emails representing approximately 84.62% and spam representing 15.38%.

The preparation process extracts:

- Email subject
- Email body
- Combined email text
- Classification label
- Source filename
- Total word count
- Character count
- Subject word count
- Body word count

The preparation script also validates message filenames, checks for email headers, handles MIME multipart messages, extracts plain-text or HTML content, and supports character encoding fallbacks.

### Data Quality

- Invalid or non-email files are filtered.
- Empty or invalid email records are rejected.
- Missing values are checked during EDA.
- Duplicate email texts are removed before model training.
- The training pipeline removes 49 normalized duplicate email texts.

## Exploratory Data Analysis

Exploratory analysis examines:

- Spam versus ham distribution.
- Class percentages.
- Missing values.
- Duplicate records and email texts.
- Email length and word-count distributions.
- Subject and body characteristics.

Visual reports are generated under:

`reports/figures/`

## Text Processing and Feature Extraction

The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to transform email text into numerical feature vectors.

Configuration:

| Parameter | Value |
|---|---|
| Maximum features | 50,000 |
| N-gram range | (1, 2) |
| Minimum document frequency | 2 |
| Maximum document frequency | 0.98 |
| Stop words | English |
| Sublinear TF | Enabled |
| Accent stripping | Unicode |

Both individual words and two-word sequences are considered, allowing the model to learn contextual patterns in email messages.

## Machine Learning Models

Three classification algorithms were trained and evaluated.

### 1. Multinomial Naive Bayes

A probabilistic classifier commonly used for text classification. It serves as a baseline model.

### 2. Logistic Regression

A linear classification algorithm that estimates the probability of an email belonging to a particular class.

### 3. Linear Support Vector Machine

A linear margin-based classifier that learns a decision boundary between spam and legitimate email messages.

### Training Configuration

- Training/testing split: 80/20
- Random state: 42
- Stratified sampling
- Five-fold stratified cross-validation
- Model selection based on mean cross-validation F1-score
- Class weighting: balanced for Logistic Regression and Linear SVM

After duplicate removal, the dataset contains 3,201 usable emails:

| Split | Ham | Spam | Total |
|---|---:|---:|---:|
| Training | 2,175 | 385 | 2,560 |
| Testing | 545 | 96 | 641 |
| **Total** | **2,720** | **481** | **3,201** |

## Model Evaluation Results

### Test Set Performance

| Model | Accuracy | Precision | Recall | F1-score | Average Precision |
|---|---:|---:|---:|---:|---:|
| Multinomial Naive Bayes | 95.32% | 100.00% | 68.75% | 81.48% | 97.91% |
| Logistic Regression | 98.44% | 93.00% | 96.88% | 94.90% | 99.35% |
| **Linear SVM** | **99.06%** | **96.88%** | **96.88%** | **96.88%** | **99.59%** |

### Cross-Validation Results

| Model | Mean CV F1 | CV F1 Standard Deviation |
|---|---:|---:|
| Multinomial Naive Bayes | 0.6634 | 0.0336 |
| Logistic Regression | 0.9101 | 0.0226 |
| **Linear SVM** | **0.9343** | **0.0187** |

**Selected model: Linear SVM**

Linear SVM achieved the highest mean cross-validation F1-score and the strongest overall test-set performance among the evaluated models.

### Confusion Matrix

The final Linear SVM confusion matrix is:

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 542 | 3 |
| Spam | 3 | 93 |

This gives:

- True Negatives: 542
- False Positives: 3
- False Negatives: 3
- True Positives: 93

The model correctly classified 635 out of 641 test emails.

## Error Analysis

An additional error-analysis stage investigates incorrectly classified emails.

### False Positives

Three legitimate emails were incorrectly classified as spam:

1. Matrox Parhelia graphics hardware announcement.
2. Netscape 7.0 browser announcement.
3. Japanese technical business correspondence.

The two English-language examples contain promotional wording, product announcements and external links. These characteristics overlap with common spam patterns.

The Japanese email demonstrates the limitations of an English-oriented TF-IDF pipeline when processing multilingual messages.

### False Negatives

Three spam emails were incorrectly classified as legitimate:

1. Sunfrom Lighting commercial product promotion.
2. Net-Temps recruitment marketing email.
3. Mercedes-Benz E240 vehicle leasing advertisement.

These messages contain commercial information, product descriptions and business contact details that resemble legitimate business correspondence.

### Error Analysis Summary

| Metric | Result |
|---|---:|
| Total test emails | 641 |
| Correct predictions | 635 |
| Incorrect predictions | 6 |
| False positives | 3 |
| False negatives | 3 |
| False-positive rate | 0.55% |
| False-negative rate | 3.13% |
| Test accuracy | 99.06% |

## Project Structure

```text
EmailSpamDetection/
│
├── dataset/
│   ├── raw/
│   │   ├── easy_ham/
│   │   ├── hard_ham/
│   │   └── spam/
│   │
│   └── processed/
│       └── emails.csv
│
├── models/
│   └── best_spam_classifier.joblib
│
├── reports/
│   ├── figures/
│   │   ├── confusion_matrices.png
│   │   ├── precision_recall_curves.png
│   │   └── model_comparison.png
│   │
│   ├── metrics/
│   │   ├── model_comparison.csv
│   │   ├── test_predictions.csv
│   │   ├── classification_reports.json
│   │   └── training_summary.json
│   │
│   └── error_analysis/
│       ├── false_positives.csv
│       ├── false_negatives.csv
│       ├── all_misclassified_emails.csv
│       └── error_analysis_summary.csv
│
└── src/
    ├── prepare_dataset.py
    ├── exploratory_analysis.py
    ├── train_spam_classifier.py
    ├── predict_spam.py
    ├── error_analysis.py
    └── inspect_false_positives.py
```

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/vnyabuto/OIBSIP.git
cd OIBSIP
```

### 2. Create and Activate a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download and extract the three Apache SpamAssassin corpus archives into:

```text
EmailSpamDetection/dataset/raw/
```

Ensure the extracted directory structure is:

```text
raw/
├── easy_ham/
├── hard_ham/
└── spam/
```

Run:

```bash
python EmailSpamDetection/src/prepare_dataset.py
```

### 5. Perform Exploratory Data Analysis

```bash
python EmailSpamDetection/src/exploratory_analysis.py
```

### 6. Train and Evaluate Models

```bash
python EmailSpamDetection/src/train_spam_classifier.py
```

### 7. Run Email Classification

```bash
python EmailSpamDetection/src/predict_spam.py
```

The prediction script loads the saved Linear SVM pipeline and accepts an email subject and body for classification.

### 8. Perform Error Analysis

```bash
python EmailSpamDetection/src/error_analysis.py
```

To inspect false positives:

```bash
python EmailSpamDetection/src/inspect_false_positives.py
```

## Technologies Used

- **Python** – Main programming language
- **Pandas** – Dataset processing and analysis
- **NumPy** – Numerical operations
- **Scikit-learn** – Machine learning and model evaluation
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Joblib** – Model serialization
- **Email Parser** – Email message extraction and MIME handling
- **Regular Expressions** – Text and filename validation

## Limitations

1. **Dataset age:** The Apache SpamAssassin corpus used in this project dates from 2003. It may not represent current spam techniques, modern phishing campaigns or contemporary email communication.

2. **Language coverage:** The TF-IDF pipeline uses English stop words and is primarily suited to English-language messages. Multilingual emails may be misclassified.

3. **Commercial email ambiguity:** Legitimate newsletters and commercial announcements can share vocabulary with unsolicited spam.

4. **Dataset size:** The corpus is relatively small compared with modern email classification datasets.

5. **Distribution shift:** High test-set performance does not guarantee equivalent performance on emails from different sources or more recent periods.

6. **Probability estimates:** Linear SVM produces a decision score rather than a calibrated spam probability.

The model should therefore be considered an educational machine learning classifier, not a complete production-grade email security or phishing detection solution.

## Future Improvements

- Introduce multilingual text processing.
- Evaluate additional contemporary email datasets.
- Experiment with advanced NLP representations and transformer-based models.
- Investigate probability calibration.
- Develop a web-based email spam detection interface.
- Add email header and sender-domain analysis.
- Evaluate robustness against modern phishing and social-engineering techniques.

## Author

**Victor Nyabuto**

GitHub: https://github.com/vnyabuto

## Internship

Oasis Infobyte Data Science Internship

**Task  – Email Spam Detection**

## License and Dataset Attribution

This project was developed for educational and internship purposes.

The email corpus is provided by the Apache SpamAssassin project. Refer to its official website for dataset information and applicable terms:

https://spamassassin.apache.org/old/publiccorpus/
