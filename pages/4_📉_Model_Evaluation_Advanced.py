import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.title("📉 Model Evaluation (Advanced)")

metrics_file = "topic_metrics.json"
if not os.path.exists(metrics_file):
    st.error("⚠️ Metrics file not found. Run train_model.py first.")
    st.stop()

with open(metrics_file, "r") as f:
    metrics = json.load(f)

# Accuracy
st.subheader("✅ Accuracy")
st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")

# Classification Report
st.subheader("📋 Classification Report")
report_df = pd.DataFrame(metrics["report"]).T
st.dataframe(report_df)

# Confusion Matrix
st.subheader("📉 Confusion Matrix")
cm = metrics["confusion_matrix"]
labels = metrics["labels"]

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
