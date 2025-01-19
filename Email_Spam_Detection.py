import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Set Streamlit page configuration for full width
st.set_page_config(page_title="Email Spam Detector", layout="wide")

# Title of the application
st.title("Email Spam Detection - [@suraj_nate](https://www.instagram.com/suraj_nate/) 👀")
st.markdown("A machine learning application to detect spam emails using a pre-trained model.")

# Loading the pre-trained model and vectorizer
try:
    model = joblib.load("spam_detector_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

except FileNotFoundError:
    st.error("Model or vectorizer not found. Ensure `spam_detector_model.pkl` and `vectorizer.pkl` are in the same directory.")
    st.stop()

# Initialize NLTK tools
nltk.download('stopwords')
ps = PorterStemmer()

# Function to clean and preprocess text
def clean_text(text):
    """Clean and preprocess text."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Section for input and prediction
st.header("Test the Spam Detector")

# Input for custom email message
input_message = st.text_area("Enter an email message:", height=150)

if st.button("Predict"):
    if input_message.strip():
        # Clean and vectorize the input message
        cleaned_message = clean_text(input_message)
        vectorized_message = vectorizer.transform([cleaned_message]).toarray()

        # Predict using the pre-trained model
        prediction = model.predict(vectorized_message)
        result = "Spam" if prediction == 1 else "Not Spam"

        # Display the result
        st.subheader("Prediction Result")
        st.success(f"The email is classified as: **{result}**")
    else:
        st.error("Please enter a message to classify.")

# Instructions for users
st.info("Note: This application uses a pre-trained Naive Bayes model to classify emails as Spam or Not Spam.")


# Footer
st.write("---")
st.markdown('<center><a href="https://www.instagram.com/suraj_nate/" target="_blank" style="color:white;text-decoration:none">&copy; 2025 @suraj_nate All rights reserved.</a></center>', unsafe_allow_html=True)
