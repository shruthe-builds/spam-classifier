"""
predict.py
----------
Loads the saved model + vectorizer and classifies new email/SMS text
as spam or ham. Can be used as a CLI (interactive loop) or imported.

Run:  python predict.py
"""

import joblib
from preprocess import clean_text

MODEL_PATH = "model/spam_classifier.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def classify(text: str, model=None, vectorizer=None) -> dict:
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = round(float(model.predict_proba(vec)[0][pred]), 4)

    return {
        "label": "spam" if pred == 1 else "ham",
        "confidence": confidence,
    }


def main():
    print("Loading model...")
    model, vectorizer = load_artifacts()
    print("Email Spam Classifier ready. Type a message and press Enter.")
    print("(Type 'quit' to exit)\n")

    while True:
        text = input("Enter email/message text: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue

        result = classify(text, model, vectorizer)
        conf_str = f" (confidence: {result['confidence']*100:.1f}%)" if result["confidence"] else ""
        print(f"  -> Prediction: {result['label'].upper()}{conf_str}\n")


if __name__ == "__main__":
    main()
