
import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("Sentiment Analysis Web App")
st.write("Enter a sentence, and the app will predict whether the sentiment is Positive or Negative.")

# Text input
text_input = st.text_area("Your Input Text")

# Predict button
if st.button("Predict Sentiment"):
    if text_input.strip():
        input_vector = vectorizer.transform([text_input])
        prediction = model.predict(input_vector)[0]
        st.success(f"Predicted Sentiment: **{prediction}**")
    else:
        st.warning("Please enter some text to analyze.")
