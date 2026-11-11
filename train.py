"""
train.py
--------
Trains and compares three models (Naive Bayes, Logistic Regression, SVM)
on the cleaned spam dataset, prints evaluation metrics for each, saves the
best-performing model + vectorizer to disk, and saves a spam-keyword
visualization.

Run:  python train.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from preprocess import clean_text

DATA_PATH = "data/spam.csv"
MODEL_PATH = "model/spam_classifier.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python download_data.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean_message"] = df["message"].apply(clean_text)
    return df


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

    return {"name": name, "model": model, "accuracy": acc, "f1": f1, "preds": preds}


def plot_confusion_matrix(y_test, preds, name):
    os.makedirs("outputs", exist_ok=True)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrix_{name.replace(' ', '_')}.png")
    plt.close()


def plot_top_spam_keywords(vectorizer, model, top_n=20):
    """Works for models exposing feature_log_prob_ (Naive Bayes) or
    coef_ (Logistic Regression / linear SVM)."""
    os.makedirs("outputs", exist_ok=True)
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "feature_log_prob_"):
        spam_scores = model.feature_log_prob_[1]
    elif hasattr(model, "coef_"):
        coef = model.coef_
        spam_scores = coef.toarray()[0] if hasattr(coef, "toarray") else coef[0]
    else:
        return

    top_idx = spam_scores.argsort()[-top_n:][::-1]
    top_words = [feature_names[i] for i in top_idx]
    top_vals = [spam_scores[i] for i in top_idx]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_vals, y=top_words, hue=top_words, legend=False, palette="Reds_r")
    plt.title("Top Spam-Indicating Keywords")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("outputs/top_spam_keywords.png")
    plt.close()


def main():
    os.makedirs("model", exist_ok=True)
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="linear", probability=True),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        result = evaluate(name, model, X_test_vec, y_test)
        plot_confusion_matrix(y_test, result["preds"], name)
        results.append(result)

    best = max(results, key=lambda r: r["f1"])
    print(f"\n>>> Best model: {best['name']} (F1 = {best['f1']:.4f}) — saving this one.")

    joblib.dump(best["model"], MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")

    plot_top_spam_keywords(vectorizer, best["model"])
    print("Saved visualizations to outputs/")


if __name__ == "__main__":
    main()
