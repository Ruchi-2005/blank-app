import streamlit as st
import pandas as pd
import joblib
import os
from custom_transformers import TextCleaner
import plotly.express as px  # Use Plotly for colored bars

st.title("🔍 Topic Sentiment Prediction")

model_path = "topic_sentiment_model.joblib"
dataset_path = "balanced_sentiment_dataset.csv"

if not os.path.exists(model_path):
    st.error("⚠️ Model not found. Run train_model.py first.")
    st.stop()

model = joblib.load(model_path)

# Load dataset
df = pd.read_csv(dataset_path)

# --- Rename columns to lowercase to match code ---
df = df.rename(columns={
    "Topic": "topic",
    "Text": "text",
    "Sentiment": "label"
})

topics = sorted(df["topic"].unique())

selected_topic = st.selectbox("Select a Topic", topics)

if st.button("Predict Sentiment for Topic"):
    topic_data = df[df["topic"] == selected_topic]
    sentences = topic_data["text"].tolist()
    predictions = model.predict(sentences)
    topic_data["Predicted"] = predictions

    # --- Color only the letters in 'Predicted' column ---
    def style_sentiment(val):
        color_map = {
            "positive": "green",
            "negative": "red",
            "neutral": "blue"
        }
        return f"color: {color_map.get(val, 'black')}; font-weight: bold; text-align: center"

    st.dataframe(topic_data.head(15).style.applymap(style_sentiment, subset=["Predicted"]))

    # -----------------------------
    # Custom-colored bar chart
    # -----------------------------
    st.subheader("📊 Sentiment Distribution")

    sentiment_counts = topic_data["Predicted"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    # Color map for bars
    color_map = {
        "positive": "#2ECC71",  # green
        "negative": "#E74C3C",  # red
        "neutral": "#3498DB"    # blue
    }

    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map=color_map,
        text="Count",
        title=f"Sentiment Distribution for '{selected_topic}'"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title="Number of Sentences", xaxis_title="Sentiment", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Overall sentiment with adaptive text color
    overall = topic_data["Predicted"].value_counts().idxmax()
    color_map_overall = {
        "positive": "green",
        "negative": "red",
        "neutral": "blue"
    }

    st.markdown(
        f"""
        <p style="font-size:18px; color:var(--text-color);">
            Overall Sentiment for '{selected_topic}': 
            <span style="color:{color_map_overall[overall]}; font-weight:bold;">{overall.upper()}</span>
        </p>
        """,
        unsafe_allow_html=True
    )
