import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def clean_text(text):
    text = str(text).lower()
    text = "".join([char for char in text if char not in string.punctuation])
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

model = None
vectorizer = None

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

def predict_spam(text):
    """
    Predicts if text is Spam / Not Spam using standard Naive Bayes, and returns confidence.
    """
    if model is None or vectorizer is None:
        raise FileNotFoundError("Model or vectorizer not found. Please run model.py first to train.")

    cleaned_text = clean_text(text)
    vectorized_input = vectorizer.transform([cleaned_text])
    
    # Predict probabilities [[prob_not_spam, prob_spam]]
    probabilities = model.predict_proba(vectorized_input)[0]
    prediction_idx = 1 if probabilities[1] > 0.5 else 0
    confidence = float(probabilities[0])  # User requested confidence to act as a Safety metric
    
    return {
        "prediction": "Spam" if prediction_idx == 1 else "Not Spam",
        "confidence": round(confidence, 4)
    }
