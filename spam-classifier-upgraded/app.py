"""Flask app: dashboard, live prediction and model evaluation.  Run: python app.py"""
import json
import threading
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

from spam_classifier import DATA_DIR, SpamEngine, ensure_trained

app = Flask(__name__)
HISTORY = DATA_DIR / "history.jsonl"
NAMES = {"naive_bayes": ("Naive Bayes", "TF-IDF + Multinomial Naive Bayes"),
         "lstm": ("LSTM", "Embedding + LSTM (Keras)")}
_engine, _lock = None, threading.Lock()


def engine():
    global _engine
    if _engine is None:
        try:
            _engine = SpamEngine()
        except Exception:  # models missing or unreadable
            return None
    return _engine


def not_ready():
    return jsonify(error="Models are not trained yet. Run: python spam_classifier.py"), 503


def primary_key(metrics):
    """The model with the best test-set F1."""
    return max(metrics["models"], key=lambda k: metrics["models"][k]["f1"])


def class_report(m):
    if "tp" not in m:
        return None
    tp, tn, fp, fn = m["tp"], m["tn"], m["false_alarms"], m["missed_spam"]
    div = lambda a, b: round(a / b, 4) if b else 0.0  # noqa: E731

    def row(p, r, support):
        return {"precision": p, "recall": r, "f1": div(2 * p * r, p + r), "support": support}

    return {"legitimate": row(div(tn, tn + fn), div(tn, tn + fp), tn + fp),
            "spam": row(div(tp, tp + fp), div(tp, tp + fn), tp + fn)}


def read_history(limit=50):
    if not HISTORY.exists():
        return []
    rows = []
    for line in HISTORY.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def metrics_payload(eng):
    m = eng.metrics
    models = []
    for key, vals in m["models"].items():
        models.append({"key": key, "name": NAMES[key][0], "algorithm": NAMES[key][1],
                       "metrics": vals, "report": class_report(vals)})
    return {"dataset": m["dataset"], "primary": primary_key(m), "models": models,
            "learning_curve": m.get("learning_curve", []), "lstm_available": m.get("lstm_available", False)}


@app.get("/")
def dashboard():
    return render_template("dashboard.html", page="dashboard", title="Dashboard")


@app.get("/predict")
def predict_page():
    return render_template("predict.html", page="predict", title="Detect spam")


@app.get("/evaluation")
def evaluation_page():
    return render_template("evaluation.html", page="evaluation", title="Model evaluation")


@app.get("/api/model-metrics")
def model_metrics():
    eng = engine()
    return jsonify(metrics_payload(eng)) if eng else not_ready()


@app.get("/api/dashboard")
def dashboard_data():
    eng = engine()
    if not eng:
        return not_ready()
    hist = read_history(200)
    data = metrics_payload(eng)
    data.update(top_words=eng.metrics["top_words"], history=hist[-50:],
                predictions={"total": len(hist), "spam": sum(h["label"] == "SPAM" for h in hist)})
    return jsonify(data)


@app.get("/api/history")
def history():
    return jsonify(read_history(10)[::-1])


@app.post("/api/predict")
def predict():
    eng = engine()
    if not eng:
        return not_ready()
    text = str((request.get_json(silent=True) or {}).get("text", "")).strip()[:1000]
    if not text:
        return jsonify(error="Enter a message to analyze."), 400
    try:
        res = eng.predict(text)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:  # noqa: BLE001
        app.logger.exception("prediction failed")
        return jsonify(error="Prediction failed. Please try again."), 500

    key = primary_key(eng.metrics)
    prob = res[key]["probability"] if res.get(key) else res["naive_bayes"]["probability"]
    res.update(primary=key, spam_probability=prob, label="SPAM" if prob >= 0.5 else "LEGITIMATE")
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "text": text[:140],
             "label": res["label"], "spam_probability": round(prob, 4)}
    with _lock:
        DATA_DIR.mkdir(exist_ok=True)
        with HISTORY.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    return jsonify(res)


if __name__ == "__main__":
    ensure_trained()
    app.run(host="127.0.0.1", port=5000, debug=False)
