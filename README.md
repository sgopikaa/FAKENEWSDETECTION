# VERASCAN — Fake News Detector Frontend

A Flask web app wrapped around the Random Forest + TF-IDF model trained in
`fakenewsdetection.ipynb`.

## Files in this bundle

```
app.py                     Flask backend (loads the model, exposes /predict)
requirements.txt           Python dependencies
templates/index.html       The page
static/css/style.css       Styling
static/js/script.js        Scan animation + API calls
```

Your existing `fake_news_model.joblib` and `tfidf_vectorizer.joblib` are
**not included** — copy your real ones from your VS Code project into this
folder before running (same level as `app.py`).

## Set up

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

The first run downloads the NLTK stopwords corpus automatically if it isn't
already cached (this needs internet access once).

## How it works

`app.py` re-implements the exact preprocessing from the notebook
(lowercase → strip non-letters → drop stopwords → Porter-stem), transforms
the cleaned text with the saved `TfidfVectorizer`, and runs it through the
saved Random Forest. `/predict` returns JSON:

```json
{
  "label": "Real",
  "prediction": 1,
  "confidence": 0.93,
  "proba_real": 0.93,
  "proba_fake": 0.07,
  "word_count": 42,
  "token_count": 27
}
```

The frontend uses that response to swing a gauge needle and reveal the
verdict — the "How it reads" and "Model bench" sections show your actual
notebook numbers (99.7% held-out accuracy, 44,689 training articles, the
four-model comparison), so update those in `index.html` if you retrain
with different data or figures.

## Notes

- Add your own favicon / OG image if you want it shareable.
- `debug=True` in `app.py` is fine for local dev — turn it off before any
  real deployment, and run behind a proper WSGI server (gunicorn, etc).
