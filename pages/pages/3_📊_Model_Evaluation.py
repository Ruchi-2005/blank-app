import streamlit as st
import json
import pandas as pd
import os

st.title("📊 Model Evaluation (Basic)")

metrics_file = "topic_metrics.json"
if not os.path.exists(metrics_file):
    st.error("⚠️ Metrics file not found. Run train_model.py first.")
    st.stop()

with open(metrics_file, "r") as f:
    metrics = json.load(f)

st.subheader("Accuracy")
st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")

st.subheader("Classification Report")
report_df = pd.DataFrame(metrics["report"]).T
st.dataframe(report_df)
