import re
import os
import joblib
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_news_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")

# Make sure the stopwords corpus is available (no-op if already downloaded)
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# label convention used at training time: 0 = Fake, 1 = Real
LABELS = {0: "Fake", 1: "Real"}


# ------------------------------------------------------------------
# Preprocessing — must mirror the training notebook exactly
# ------------------------------------------------------------------
def preprocess(text: str) -> str:
    text = str(text)
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    words = text.split()
    words = [STEMMER.stem(word) for word in words if word not in STOP_WORDS]
    return " ".join(words)


def analyze(raw_text: str) -> dict:
    cleaned = preprocess(raw_text)
    vector = vectorizer.transform([cleaned])

    prediction = int(model.predict(vector)[0])
    label = LABELS.get(prediction, "Unknown")

    confidence = None
    proba_real = None
    proba_fake = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        # class order follows model.classes_
        class_index = {cls: i for i, cls in enumerate(model.classes_)}
        proba_fake = float(proba[class_index.get(0, 0)])
        proba_real = float(proba[class_index.get(1, 1)])
        confidence = max(proba_fake, proba_real)

    word_count = len(raw_text.split())
    token_count = len(cleaned.split())

    return {
        "label": label,
        "prediction": prediction,
        "confidence": confidence,
        "proba_real": proba_real,
        "proba_fake": proba_fake,
        "word_count": word_count,
        "token_count": token_count,
    }


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please paste some article text to analyze."}), 400

    if len(text.split()) < 5:
        return jsonify({"error": "Please paste a longer excerpt (at least a few sentences) for a reliable reading."}), 400

    try:
        result = analyze(text)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Something went wrong during analysis: {exc}"}), 500

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
