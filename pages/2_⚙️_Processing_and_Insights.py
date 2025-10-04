import streamlit as st
import pandas as pd
import altair as alt

st.title("⚙️ Processing & Insights")

df = pd.read_csv("balanced_sentiment_dataset.csv")

# --- Rename columns to lowercase to match code ---
df = df.rename(columns={
    "Topic": "topic",
    "Text": "text",
    "Sentiment": "label"
})

st.markdown("### Dataset Overview")

# Function to style the 'label' column
def style_sentiment(val):
    color_map = {
        "positive": "green",
        "negative": "red",
        "neutral": "blue"
    }
    return f"color: {color_map.get(val, 'black')}; font-weight: bold"

# Show all rows in a scrollable table with styled 'label' column
st.dataframe(df.style.applymap(style_sentiment, subset=["label"]))

# -----------------------------
# Topic Distribution with purple color
# -----------------------------
st.markdown("### Topic Distribution")
topic_counts = df["topic"].value_counts().reset_index()
topic_counts.columns = ["topic", "count"]

topic_chart = alt.Chart(topic_counts).mark_bar(color="#0ED6F1").encode(
    x=alt.X("topic:N", sort="-y", title="Topic"),
    y=alt.Y("count:Q", title="Count")
).properties(width=700, height=400)

st.altair_chart(topic_chart)

# -----------------------------
# Sentiment Distribution with custom colors
# -----------------------------
st.markdown("### Sentiment Distribution")
sentiment_counts = df["label"].value_counts().reset_index()
sentiment_counts.columns = ["label", "count"]

sentiment_chart = alt.Chart(sentiment_counts).mark_bar().encode(
    x=alt.X("label:N", title="Sentiment"),
    y=alt.Y("count:Q", title="Count"),
    color=alt.Color("label:N",
                    scale=alt.Scale(domain=["positive", "negative", "neutral"],
                                    range=["green", "red", "blue"]))
).properties(width=700, height=400)

st.altair_chart(sentiment_chart)

st.info("✅ Dataset is balanced across positive, negative, and neutral sentiments.")
