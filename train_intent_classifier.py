"""
Step 4 (v3): choose the intent classifier by 5-FOLD CROSS-VALIDATION,
then retrain the winner on ALL training data and save it.

Why cross-validation replaced the old single 80/20 split:
with only ~30 patterns per intent, one random split leaves ~6 test
examples per intent, so which model "wins" depends heavily on luck of
the split (the old script crowned Naive Bayes, which then scored 0/5
on exercise_by_level questions on the independent shared test set).
5-fold CV trains/tests on 5 different splits and averages macro-F1,
giving a far more stable estimate of each model's true ability. The
winner is then retrained on 100% of the data (no data wasted on a
held-out split) because the REAL evaluation happens later on the
separate shared_test_set.csv, which no model ever trains on.

Run:  python train_intent_classifier.py
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ---------------------------------------------------------------- load data
with open("intents.json") as f:
    data = json.load(f)

texts, labels = [], []
for intent in data["intents"]:
    for pattern in intent.get("patterns", []):
        texts.append(pattern)
        labels.append(intent["tag"])
texts, labels = np.array(texts), np.array(labels)

print(f"Total training examples: {len(texts)}")
print(pd.Series(labels).value_counts().to_string(), "\n")

# ----------------------------------------------- candidate model pipelines
# Each pipeline bundles TF-IDF + classifier so that, inside every CV fold,
# the vectorizer is fitted ONLY on that fold's training part (no leakage).
def make_features():
    """Word features capture meaning ("workout plan"); CHARACTER features
    capture spelling ("heyy" shares the chunks "hey"/"eyy" with "hey"), so
    typos and unseen word forms still land near the right intent. Combining
    both raised CV macro-F1 from 0.741 to 0.831."""
    return FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb",
                                 ngram_range=(3, 5))),
    ])


def make_pipeline(model):
    return Pipeline([("feats", make_features()), ("clf", model)])

models = {
    "Naive Bayes":         make_pipeline(MultinomialNB()),
    "Linear SVM":          make_pipeline(CalibratedClassifierCV(LinearSVC())),
    "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000)),
}

# ------------------------------------------------- 5-fold stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
print("=" * 62)
print(f"{'Model':<22}{'CV macro-F1 (mean)':>20}{'(std)':>10}")
print("-" * 62)
for name, pipe in models.items():
    scores = cross_val_score(pipe, texts, labels, cv=cv, scoring="f1_macro")
    results[name] = scores
    print(f"{name:<22}{scores.mean():>20.3f}{scores.std():>10.3f}")
print("=" * 62)

best_name = max(results, key=lambda n: results[n].mean())
print(f"\nBest model by mean CV macro-F1: {best_name}")
print(f"Per-fold scores: {np.round(results[best_name], 3).tolist()}\n")

# --------------------------- retrain winner on ALL data, save components
vectorizer = make_features()
X_full = vectorizer.fit_transform(texts)
winner = {
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": CalibratedClassifierCV(LinearSVC()),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}[best_name]
winner.fit(X_full, labels)

with open("intent_classifier.pkl", "wb") as f:
    pickle.dump({"vectorizer": vectorizer, "model": winner,
                 "model_name": best_name}, f)
print(f"Retrained {best_name} on all {len(texts)} examples "
      f"and saved to intent_classifier.pkl")
