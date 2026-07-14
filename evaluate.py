"""
Step 7: evaluate the WHOLE bot (classifier + confidence threshold) on the
shared held-out test set. Your teammate runs the same CSV through their
Rasa/Dialogflow bot and you compare the numbers side by side.

Run in the same folder as shared_test_set.csv + intent_classifier.pkl:
    python evaluate.py

Outputs:
1. Overall accuracy on the shared test set
2. Per-intent precision / recall / F1 (paste into report)
3. confusion_matrix.png  -> insert into Results & Discussion
4. eval_predictions.csv  -> every question with predicted vs expected,
   so you can discuss WHICH mistakes the bot makes, not just how many.
"""

import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)

CONFIDENCE_THRESHOLD = 0.14  # keep identical to chatbot_core.py

with open("intent_classifier.pkl", "rb") as f:
    saved = pickle.load(f)
vectorizer, clf = saved["vectorizer"], saved["model"]
model_name = saved.get("model_name", "classifier")

test = pd.read_csv("shared_test_set.csv")

from chatbot_core import chat  # evaluate the REAL bot, including the
                                # rule-based override in chat()

def predict(text):
    intent, confidence, slots, reply = chat(text)
    return intent, confidence

test[["predicted_intent", "confidence"]] = test["text"].apply(
    lambda t: pd.Series(predict(t))
)
test["correct"] = test["predicted_intent"] == test["expected_intent"]

acc = accuracy_score(test["expected_intent"], test["predicted_intent"])
print(f"Model: {model_name}   |   Shared test set: {len(test)} questions")
print(f"Overall accuracy: {acc:.1%}\n")

print(classification_report(test["expected_intent"], test["predicted_intent"],
                            zero_division=0))

print("--- Misclassified questions ---")
wrong = test[~test["correct"]]
if wrong.empty:
    print("(none)")
for _, r in wrong.iterrows():
    print(f"  {r['text']!r}: expected {r['expected_intent']}, "
          f"got {r['predicted_intent']} ({r['confidence']:.2f})")

labels = sorted(test["expected_intent"].unique())
cm = confusion_matrix(test["expected_intent"], test["predicted_intent"],
                      labels=labels)
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay(cm, display_labels=labels).plot(
    ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
ax.set_title(f"Intent Confusion Matrix — {model_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nSaved confusion_matrix.png")

test.to_csv("eval_predictions.csv", index=False)
print("Saved eval_predictions.csv")
