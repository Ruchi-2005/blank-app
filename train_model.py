import pandas as pd, joblib, json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from custom_transformers import TextCleaner

# Load dataset
df = pd.read_csv("balanced_sentiment_dataset.csv")

# --- Rename columns to match your code ---
df = df.rename(columns={
    "Topic": "topic",
    "Text": "text",
    "Sentiment": "label"
})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Pipeline
pipeline = Pipeline([
    ("cleaner", TextCleaner()),
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=200))
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred, labels=["negative", "neutral", "positive"])

# Save model and metrics
joblib.dump(pipeline, "topic_sentiment_model.joblib")
with open("topic_metrics.json", "w") as f:
    json.dump({"accuracy": acc, "report": report, "confusion_matrix": cm.tolist(), "labels": ["negative","neutral","positive"]}, f, indent=4)

print("✅ Model trained! Accuracy:", acc)
