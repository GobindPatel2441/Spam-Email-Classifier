# 📧 Spam Email Classifier

A beginner-friendly full-stack machine learning application that classifies email text as **Spam** or **Not Spam** — complete with a real-time **Safety Confidence Gauge**.

Built with Python (Flask), Scikit-learn (Naive Bayes + TF-IDF), and a clean HTML/JS frontend.

---

## 🎯 How It Works

1. **User Input:** Paste any email or message text into the frontend UI.
2. **API Request:** JavaScript sends a `POST` request to the Flask backend.
3. **ML Prediction:** The Flask API cleans the text (lowercase, stopword removal via NLTK, stemming), vectorizes it with TF-IDF, and runs it through a Multinomial Naive Bayes model.
4. **Response:** Returns a prediction (`"Spam"` or `"Not Spam"`) and a **Safety Confidence** score (0–100%).
5. **UI Display:** A gradient confidence gauge updates dynamically — needle points left for Spam (🔴), right for Safe (🟢).

---

## 🧠 Why Naive Bayes?

- **Fast & Efficient:** Calculates conditional probabilities based on word frequencies.
- **Great for Text:** TF-IDF vectorization + Naive Bayes is a proven combination for spam detection.
- **Beginner Friendly:** Easy to understand, train, and extend.

---

## 📁 Project Structure

```
Spam-Email-Classifier/
│
├── project/
│   ├── backend/
│   │   ├── app.py          → Flask API Server
│   │   ├── model.py        → Training script (Naive Bayes)
│   │   ├── predictor.py    → Inference logic (predict_spam)
│   │   ├── dataset.csv     → SMS Spam Collection training data
│   │   ├── model.pkl       → Trained model weights (auto-generated)
│   │   └── vectorizer.pkl  → TF-IDF Vectorizer (auto-generated)
│   │
│   ├── frontend/
│   │   └── index.html      → Full UI with confidence gauge
│   │
│   ├── requirements.txt    → Python dependencies
│   └── README.md
│
└── .gitignore
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r project/requirements.txt
```

### 2. Train the Model *(only if model.pkl is missing)*
```bash
cd project/backend
python model.py
```

### 3. Start the Flask Backend
```bash
cd project/backend
python app.py
```
> Backend runs at `http://localhost:5000`

### 4. Start the Frontend Server
```bash
cd project/frontend
python -m http.server 8080
```
> Open `http://localhost:8080/index.html` in your browser

---

## 🖥️ UI Features

| Feature | Description |
|---|---|
| 🚨 **Spam Detected** | Low confidence (0–50%) — DANGER ZONE |
| ⚠️ **Possible Scam** | Medium confidence (50–90%) — SUSPICIOUS |
| ✅ **Not Spam — Safe** | High confidence (90–100%) — LOOKS SAFE |
| 🎚️ **Confidence Gauge** | Animated needle on a gradient bar (Red → Yellow → Green) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS (Tailwind), JavaScript |
| Backend | Python, Flask, Flask-CORS |
| ML Model | Scikit-learn (MultinomialNB, TfidfVectorizer) |
| NLP | NLTK (stopwords, PorterStemmer) |
| Dataset | SMS Spam Collection (UCI) |
