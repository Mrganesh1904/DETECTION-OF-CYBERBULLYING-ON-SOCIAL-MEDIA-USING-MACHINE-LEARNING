from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def clean_text(text: str) -> str:
    """Basic normalization suitable for a simple educational NLP baseline."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    labels = set(df["label"].unique())
    if not labels.issubset({0, 1}) or len(labels) < 2:
        raise ValueError("label must contain both binary classes: 0 and 1.")

    if len(df) < 10:
        raise ValueError("Dataset is too small. Provide at least 10 valid rows.")

    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a cyberbullying text classifier.")
    parser.add_argument("--data", default="data/sample_data.csv", help="CSV dataset path.")
    parser.add_argument("--model", default="model.joblib", help="Output model path.")
    args = parser.parse_args()

    data_path = Path(args.data)
    df = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )

    print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, digits=4, zero_division=0))

    model_path = Path(args.model)
    joblib.dump(model, model_path)
    print(f"\nSaved trained model to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
