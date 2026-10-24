from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from train import clean_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict cyberbullying for one text message.")
    parser.add_argument("--text", required=True, help="Text to classify.")
    parser.add_argument("--model", default="model.joblib", help="Trained model path.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist. Train the model first with src/train.py."
        )

    model = joblib.load(model_path)
    cleaned = clean_text(args.text)
    prediction = int(model.predict([cleaned])[0])
    probabilities = model.predict_proba([cleaned])[0]
    bullying_probability = float(probabilities[list(model.classes_).index(1)])

    label = "CYBERBULLYING" if prediction == 1 else "NOT CYBERBULLYING"
    print(f"Prediction: {label}")
    print(f"Cyberbullying probability: {bullying_probability:.3f}")


if __name__ == "__main__":
    main()
