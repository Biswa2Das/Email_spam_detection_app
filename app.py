import streamlit as st
import pickle
import re
import string

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    with open("spam_classifier.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, vectorizer = load_model()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Streamlit App UI
st.title("📧 Spam Email Detector")
st.write("Enter an email below to check if it's spam or not.")

email_text = st.text_area("Email Content:", height=200)

if st.button("Check Spam"):
    if email_text.strip():
        cleaned_text = preprocess_text(email_text)
        email_vector = vectorizer.transform([cleaned_text])  # Convert to TF-IDF
        prediction = model.predict(email_vector)[0]
        result = "🚨 Spam" if prediction == 1 else "✅ Not Spam"
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("Please enter an email message.")

