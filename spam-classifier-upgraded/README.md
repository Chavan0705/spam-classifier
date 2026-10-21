# SpamGuard — Spam Classifier Dashboard

A Flask web app for comparing classical machine learning with deep learning on the SMS Spam Collection.

## Current working models
- **Naive Bayes** — TF-IDF + Multinomial Naive Bayes.
- **LSTM** — optional TensorFlow/Keras sequence model (when TensorFlow is installed).

The `models/bert_model.py` and `models/prompt_classifier.py` files are extension points only. BERT and prompt/LLM classification are **not active models** in this build, and the dashboard does not invent metrics for them. They require their own implementation and evaluation before being enabled.

## Run on Windows (VS Code)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. On first run, the app downloads the SMS Spam Collection and trains available models. If TensorFlow installation is troublesome, the app can run with Naive Bayes only; use `python spam_classifier.py --no-dl` to train only that model.

## Pages
- `/` — interactive dashboard
- `/predict` — detect spam in a message
- `/evaluation` — model evaluation

## API
- `GET /api/dashboard`
- `GET /api/model-metrics`
- `GET /api/history`
- `POST /api/predict` with JSON `{"text": "your message"}`

## Project layout

- `app.py` — Flask routes and API
- `spam_classifier.py` — dataset loading, training, evaluation and prediction engine
- `models/` — model adapter/extension modules
- `templates/` — page templates (including requested `index.html`, `detect.html`, `compare.html`, and `dashboard.html`)
- `static/css/style.css` — styling
- `static/js/` — interactive dashboard and prediction scripts
- `data/` — runtime dataset/model history used by the current trainer
- `dataset/` — dataset notes/extension area
- `notebooks/` — optional experiments

The current training engine reads/writes runtime data in `data/` for compatibility. Put the downloaded dataset there as `SMSSpamCollection` or `spam.csv` (`v1` and `v2` columns).

## Retraining

```powershell
python spam_classifier.py
python spam_classifier.py --no-dl
python spam_classifier.py --no-curve
```

Charts use saved test-set evaluation metrics and prediction history. Model probabilities are raw scores and should not be interpreted as calibrated probabilities without further validation.
