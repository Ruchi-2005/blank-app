# app.py

import streamlit as st

st.set_page_config(
    page_title="Topic Sentiment Dashboard",
    page_icon="☘️",
    layout="wide"
)

st.title("Modern Topic Sentiment Analyzer")
st.markdown("""
### Welcome!
Explore and analyze worldwide trending topics and their sentiments.

**Navigation:**
- 🔍 **Topic Sentiment Prediction** → Predict sentiment for any topic.
- ⚙️ **Processing & Insights** → View how data and predictions are processed.
- 📊 **Model Evaluation** → Check model metrics, accuracy, and confusion matrix.
""")

st.success("Use the sidebar or top navigation to explore the app!")
