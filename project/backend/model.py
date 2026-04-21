import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    text = str(text).lower()
    text = "".join([char for char in text if char not in string.punctuation])
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

def train_and_save_model():
    print("Loading SMS Spam Dataset...")
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    df = pd.read_csv(dataset_path, encoding="latin-1")
    
    df = df[['v1', 'v2']]
    df.columns = ['label', 'messages']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print("Cleaning data...")
    df['cleaned_messages'] = df['messages'].apply(clean_text)
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_messages'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")
    
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("Model and Vectorizer saved to backend folder!")

if __name__ == "__main__":
    train_and_save_model()
