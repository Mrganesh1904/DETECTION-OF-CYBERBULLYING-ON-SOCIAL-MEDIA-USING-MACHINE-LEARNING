# Cyberbullying Detection on Social Media Using Machine Learning

A reproducible educational machine-learning project for classifying social-media text as **cyberbullying** or **not cyberbullying**.

> This repository is a newly built implementation inspired by my undergraduate academic project topic. It is not presented as the original source code used during my degree.

## Features
- Cleans social-media text
- Converts text to TF-IDF features
- Trains Logistic Regression
- Prints accuracy, precision, recall, F1-score, confusion matrix, and classification report
- Saves the trained pipeline
- Predicts labels for new text

## Project structure
```text
cyberbullying-detection-ml/
├── data/
│   └── sample_data.csv
├── src/
│   ├── train.py
│   └── predict.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset format
Use a CSV containing:
- `text`: social-media message
- `label`: `0` for not cyberbullying, `1` for cyberbullying

The included `sample_data.csv` is a small synthetic demo dataset only. For meaningful evaluation, replace it with a larger, properly licensed dataset.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

## Train
```bash
python src/train.py --data data/sample_data.csv
```

The model is saved to `model.joblib`.

## Predict
```bash
python src/predict.py --text "You are useless and nobody likes you"
```

Or:
```bash
python src/predict.py --text "Great work on your presentation today"
```

## Important
Do not claim a particular accuracy in a CV or university application unless you obtained that result yourself on a documented dataset and evaluation split.

## Suggested CV wording
**Cyberbullying Detection on Social Media Using Machine Learning — Undergraduate Academic Project**

Developed an academic machine-learning project focused on detecting cyberbullying in social-media text. Subsequently created a reproducible Python implementation using text preprocessing, TF-IDF feature extraction, Logistic Regression, and standard classification metrics to further develop practical understanding of NLP and machine-learning workflows.
