"""
Topic 9: Spam Classifier - Classical ML vs Deep Learning
=========================================================

Everything about the models lives in this file.

    Step 1  load_dataset(), tokenize()            load + clean the text
    Step 2  train_naive_bayes()                   TF-IDF + Multinomial Naive Bayes
    Step 3  train_lstm()                          Embedding + LSTM (Keras)
    Step 4  train_all()                           comparison table + learning curve
    Step 5  SpamEngine.explain_*()                why each model made its call
    Step 6  discussion lives in the web app (templates/index.html)

Run it:
    python spam_classifier.py              # trains both models, prints the comparison
    python spam_classifier.py --no-dl      # Naive Bayes only (no TensorFlow needed)
    python spam_classifier.py --no-curve   # skip the learning-curve experiment (faster)
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"

SEED = 42
MAX_LEN = 60          # LSTM sees at most this many tokens per message
VOCAB_SIZE = 6000     # LSTM vocabulary size (incl. padding and unknown)
CURVE_FRACTIONS = (0.05, 0.1, 0.25, 0.5)


# --------------------------------------------------------------------------- #
# Step 1 - load and clean
# --------------------------------------------------------------------------- #
DATA_SOURCES = [
    ("zip", "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"),
    ("tsv", "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"),
]

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_NUM_RE = re.compile(r"\d+")
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase, mask links and numbers, then split into word tokens."""
    text = str(text).lower()
    text = _URL_RE.sub(" urltoken ", text)
    text = _NUM_RE.sub(" numtoken ", text)
    return _WORD_RE.findall(text)


def _download_dataset(target: Path) -> None:
    errors = []
    for kind, url in DATA_SOURCES:
        try:
            print(f"[data] downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=30).read()
            if kind == "zip":
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    raw = zf.read("SMSSpamCollection")
            target.write_bytes(raw)
            return
        except Exception as exc:  # noqa: BLE001 - we try the next source
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        "Could not download the SMS Spam Collection.\n"
        "Download it manually from https://archive.ics.uci.edu/dataset/228/sms+spam+collection\n"
        f"and save the file 'SMSSpamCollection' into {DATA_DIR}\n"
        "(a Kaggle 'spam.csv' with columns v1/v2 in the same folder also works).\n"
        "Errors:\n  " + "\n  ".join(errors)
    )


def load_dataset() -> pd.DataFrame:
    """Return a DataFrame with columns: text, y (1 = spam, 0 = ham)."""
    DATA_DIR.mkdir(exist_ok=True)
    tsv = DATA_DIR / "SMSSpamCollection"
    kaggle = DATA_DIR / "spam.csv"

    if kaggle.exists() and not tsv.exists():
        df = pd.read_csv(kaggle, encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "text"]
    else:
        if not tsv.exists():
            _download_dataset(tsv)
        df = pd.read_csv(
            tsv, sep="\t", header=None, names=["label", "text"],
            quoting=3, encoding="utf-8", encoding_errors="replace",
        )

    df = df.dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    # Exact duplicates would leak between train and test, so keep one copy.
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["y"] = (df["label"] == "spam").astype(int)
    return df[["text", "y"]]


def prepare_split():
    """Clean the text and make one stratified 80/20 split shared by both models."""
    df = load_dataset()
    df["tokens"] = df["text"].map(tokenize)
    df = df[df["tokens"].map(len) > 0].reset_index(drop=True)

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=df["y"], random_state=SEED
    )
    tr, te = df.iloc[idx_train], df.iloc[idx_test]
    return {
        "df": df,
        "train_tokens": tr["tokens"].tolist(),
        "test_tokens": te["tokens"].tolist(),
        "y_train": tr["y"].to_numpy(),
        "y_test": te["y"].to_numpy(),
    }


# --------------------------------------------------------------------------- #
# Shared evaluation helper
# --------------------------------------------------------------------------- #
def evaluate(y_true, prob, train_time, infer_ms, n_params) -> dict:
    pred = (np.asarray(prob) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "tp": int(tp),
        "tn": int(tn),
        "prob_hist": {   # test-set spam probability, 10 bins, split by true class
            "spam": np.histogram(np.asarray(prob)[np.asarray(y_true) == 1], bins=10, range=(0, 1))[0].tolist(),
            "ham": np.histogram(np.asarray(prob)[np.asarray(y_true) == 0], bins=10, range=(0, 1))[0].tolist(),
        },
        "false_alarms": int(fp),   # ham marked as spam
        "missed_spam": int(fn),    # spam let through
        "train_seconds": round(float(train_time), 2),
        "inference_ms": round(float(infer_ms), 3),
        "parameters": int(n_params),
    }


# --------------------------------------------------------------------------- #
# Step 2 - baseline: TF-IDF + Multinomial Naive Bayes
# --------------------------------------------------------------------------- #
def make_vectorizer() -> TfidfVectorizer:
    # Text is already cleaned, so split on spaces and keep single letters ("u" matters in SMS).
    return TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                           min_df=2, sublinear_tf=True)


def train_naive_bayes(train_tokens, y_train, test_tokens, y_test):
    train_text = [" ".join(t) for t in train_tokens]
    test_text = [" ".join(t) for t in test_tokens]

    t0 = time.perf_counter()
    vec = make_vectorizer()
    x_train = vec.fit_transform(train_text)
    nb = MultinomialNB(alpha=0.1).fit(x_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    prob = nb.predict_proba(vec.transform(test_text))[:, 1]
    infer_ms = (time.perf_counter() - t0) / max(len(test_text), 1) * 1000

    n_params = nb.feature_log_prob_.size + nb.class_log_prior_.size
    return vec, nb, evaluate(y_test, prob, train_time, infer_ms, n_params), x_train


def top_nb_words(vec, nb, x_train, k=12, min_docs=5):
    """Words with the largest log-ratio between spam and ham (seen in >= min_docs messages)."""
    delta = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    doc_freq = np.asarray((x_train > 0).sum(axis=0)).ravel()
    words = np.array(vec.get_feature_names_out())
    ok = np.where(doc_freq >= min_docs)[0]
    order = ok[np.argsort(delta[ok])]
    to_row = lambda j: {"word": words[j], "weight": round(float(delta[j]), 2)}  # noqa: E731
    return {
        "spam": [to_row(j) for j in order[::-1][:k]],
        "ham": [to_row(j) for j in order[:k]],
    }


# --------------------------------------------------------------------------- #
# Step 3 - deep model: Embedding + LSTM
# --------------------------------------------------------------------------- #
def tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def build_vocab(token_lists, size=VOCAB_SIZE) -> list[str]:
    counts = Counter(t for toks in token_lists for t in toks)
    return ["<pad>", "<oov>"] + [w for w, _ in counts.most_common(size - 2)]


def encode_batch(token_lists, stoi, max_len=MAX_LEN) -> np.ndarray:
    out = np.zeros((len(token_lists), max_len), dtype="int32")  # 0 = padding
    for i, toks in enumerate(token_lists):
        ids = [stoi.get(t, 1) for t in toks[:max_len]]         # 1 = unknown word
        out[i, : len(ids)] = ids
    return out


def build_lstm(vocab_size: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(vocab_size, 32, mask_zero=True),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm(train_tokens, y_train, test_tokens, y_test, epochs=20, verbose=0):
    from tensorflow import keras

    keras.utils.set_random_seed(SEED)
    itos = build_vocab(train_tokens)
    stoi = {w: i for i, w in enumerate(itos)}
    x_train = encode_batch(train_tokens, stoi)
    x_test = encode_batch(test_tokens, stoi)

    model = build_lstm(len(itos))
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)

    t0 = time.perf_counter()
    history = model.fit(
        x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=64,
        class_weight={0: float(weights[0]), 1: float(weights[1])},
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                 restore_best_weights=True)],
        verbose=verbose,
    )
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    prob = model.predict(x_test, batch_size=256, verbose=0).ravel()
    infer_ms = (time.perf_counter() - t0) / max(len(x_test), 1) * 1000

    metrics = evaluate(y_test, prob, train_time, infer_ms, model.count_params())
    metrics["epochs_run"] = len(history.history["loss"])
    metrics["history"] = {k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()
                          if k in ("loss", "val_loss", "accuracy", "val_accuracy")}
    return model, itos, metrics


# --------------------------------------------------------------------------- #
# Step 4 - train everything, compare, save
# --------------------------------------------------------------------------- #
def learning_curve(split, use_dl):
    """F1 on the same test set when each model sees only a fraction of the training data."""
    rows = []
    n_train = len(split["y_train"])
    for frac in CURVE_FRACTIONS:
        idx, _ = train_test_split(np.arange(n_train), train_size=frac,
                                  stratify=split["y_train"], random_state=SEED)
        toks = [split["train_tokens"][i] for i in idx]
        y = split["y_train"][idx]
        row = {"fraction": frac, "n": int(len(idx))}
        row["nb_f1"] = train_naive_bayes(toks, y, split["test_tokens"], split["y_test"])[2]["f1"]
        if use_dl:
            row["lstm_f1"] = train_lstm(toks, y, split["test_tokens"], split["y_test"])[2]["f1"]
        rows.append(row)
        print(f"[curve] {int(frac * 100):>3}% of training data ({len(idx)} msgs): {row}")
    return rows


def train_all(use_dl=True, curve=True) -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    split = prepare_split()
    df = split["df"]
    use_dl = use_dl and tf_available()
    if not use_dl:
        print("[info] TensorFlow not available or disabled: training Naive Bayes only.")

    print("[1/4] data ready:", len(df), "messages,", int(df["y"].sum()), "spam")
    print("[2/4] training TF-IDF + Naive Bayes ...")
    vec, nb, nb_metrics, x_train = train_naive_bayes(
        split["train_tokens"], split["y_train"], split["test_tokens"], split["y_test"])
    joblib.dump({"vec": vec, "nb": nb}, MODEL_DIR / "naive_bayes.joblib")

    models = {"naive_bayes": nb_metrics}
    if use_dl:
        print("[3/4] training Embedding + LSTM ...")
        lstm, itos, lstm_metrics = train_lstm(
            split["train_tokens"], split["y_train"], split["test_tokens"], split["y_test"],
            verbose=2)
        lstm.save(MODEL_DIR / "lstm.keras")
        (MODEL_DIR / "vocab.json").write_text(json.dumps(itos))
        models["lstm"] = lstm_metrics

    curve_rows = []
    if curve:
        print("[4/4] learning-curve experiment ...")
        curve_rows = learning_curve(split, use_dl)
        last = {"fraction": 1.0, "n": int(len(split["y_train"])), "nb_f1": nb_metrics["f1"]}
        if use_dl:
            last["lstm_f1"] = models["lstm"]["f1"]
        curve_rows.append(last)

    result = {
        "dataset": {
            "name": "SMS Spam Collection",
            "total": int(len(df)),
            "spam": int(df["y"].sum()),
            "ham": int((1 - df["y"]).sum()),
            "spam_pct": round(float(df["y"].mean() * 100), 1),
            "train": int(len(split["y_train"])),
            "test": int(len(split["y_test"])),
        },
        "models": models,
        "top_words": top_nb_words(vec, nb, x_train),
        "learning_curve": curve_rows,
        "lstm_available": use_dl,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(result, indent=2))
    print_comparison(result)
    return result


def print_comparison(result: dict) -> None:
    m = result["models"]
    nb, dl = m["naive_bayes"], m.get("lstm")
    rows = [("Accuracy", "accuracy"), ("Precision (spam)", "precision"),
            ("Recall (spam)", "recall"), ("F1 (spam)", "f1"),
            ("False alarms", "false_alarms"), ("Missed spam", "missed_spam"),
            ("Training time (s)", "train_seconds"), ("Inference (ms/msg)", "inference_ms"),
            ("Parameters", "parameters")]
    print("\n" + "=" * 58)
    print(f"{'Metric':<22}{'Naive Bayes':>17}{'LSTM':>17}")
    print("-" * 58)
    for label, key in rows:
        print(f"{label:<22}{nb[key]:>17}{(dl[key] if dl else '-'):>17}")
    print("=" * 58)


def ensure_trained() -> None:
    needed = [MODEL_DIR / "naive_bayes.joblib", MODEL_DIR / "metrics.json"]
    if not all(p.exists() for p in needed):
        print("[setup] No trained models found - training now (first run only).")
        train_all()


# --------------------------------------------------------------------------- #
# Step 5 - inference + interpretability
# --------------------------------------------------------------------------- #
class SpamEngine:
    """Loads the saved models and explains each prediction."""

    def __init__(self):
        bundle = joblib.load(MODEL_DIR / "naive_bayes.joblib")
        self.vec, self.nb = bundle["vec"], bundle["nb"]
        self.delta = self.nb.feature_log_prob_[1] - self.nb.feature_log_prob_[0]
        self.bias = float(self.nb.class_log_prior_[1] - self.nb.class_log_prior_[0])
        self.metrics = json.loads((MODEL_DIR / "metrics.json").read_text())

        self.lstm = None
        lstm_path, vocab_path = MODEL_DIR / "lstm.keras", MODEL_DIR / "vocab.json"
        if lstm_path.exists() and vocab_path.exists() and tf_available():
            from tensorflow import keras
            self.lstm = keras.models.load_model(lstm_path)
            self.stoi = {w: i for i, w in enumerate(json.loads(vocab_path.read_text()))}

    # --- Naive Bayes: exact, word-by-word evidence ------------------------- #
    def explain_nb(self, tokens):
        row = self.vec.transform([" ".join(tokens)]).tocsr()
        weight = dict(zip(row.indices.tolist(), row.data.tolist()))
        vocab = self.vec.vocabulary_
        counts = Counter(tokens)

        scores, known = [], []
        for tok in tokens:
            j = vocab.get(tok)
            if j is None:
                scores.append(0.0)
                known.append(False)
            else:
                scores.append(weight[j] * float(self.delta[j]) / counts[tok])
                known.append(True)

        words_total = float(sum(scores))
        log_odds = self.bias + words_total
        prob = float(self.nb.predict_proba(row)[0, 1])
        return {
            "probability": prob,
            "scores": [round(s, 4) for s in scores],
            "known": known,
            "bias": round(self.bias, 3),
            "words_total": round(words_total, 3),
            "log_odds": round(log_odds, 3),
        }

    # --- LSTM: black box, probed by deleting one word at a time ------------ #
    def explain_lstm(self, tokens):
        n = min(len(tokens), MAX_LEN)
        variants = [tokens] + [tokens[:i] + tokens[i + 1:] for i in range(n)]
        probs = self.lstm(encode_batch(variants, self.stoi), training=False).numpy().ravel()
        full = float(probs[0])
        scores = [full - float(probs[i + 1]) for i in range(n)] + [0.0] * (len(tokens) - n)
        return {"probability": full, "scores": [round(s, 4) for s in scores],
                "truncated": len(tokens) > MAX_LEN}

    def predict(self, text: str) -> dict:
        tokens = tokenize(text)
        if not tokens:
            raise ValueError("Type a message with at least one word.")
        return {
            "tokens": tokens,
            "naive_bayes": self.explain_nb(tokens),
            "lstm": self.explain_lstm(tokens) if self.lstm is not None else None,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the spam classifiers.")
    parser.add_argument("--no-dl", action="store_true", help="skip the LSTM")
    parser.add_argument("--no-curve", action="store_true", help="skip the learning curve")
    args = parser.parse_args()
    train_all(use_dl=not args.no_dl, curve=not args.no_curve)
