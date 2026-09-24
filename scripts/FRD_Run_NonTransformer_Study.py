# -*- coding: utf-8 -*-
r"""
FRD Non-Transformer Comparative Study
=====================================

Purpose
-------
Re-run ALL non-Transformer models from the paper on the SAME duplicate-safe
70/10/20 FRD split used by the current Transformer pipeline, and report the
same evaluation family:

  * Accuracy, Precision, Recall, F1
  * ROC-AUC, PR-AUC
  * Specificity, Balanced Accuracy, MCC
  * FPR, FNR, TN, FP, FN, TP
  * default threshold = 0.5
  * validation-selected threshold (same rule as the Transformer code)
  * 95% bootstrap CIs for Accuracy, F1, ROC-AUC
  * saved validation/test probabilities for paired significance tests

Models
------
Traditional ML:
  1. Naive Bayes (bag-of-words)
  2. Logistic Regression (bag-of-words)
  3. TF-IDF + Logistic Regression + POS features
  4. TF-IDF + Logistic Regression with word n-grams
  5. Word2Vec + Random Forest
  6. TF-IDF + linear SVM (train-only probability calibration)

Deep Learning:
  7. LSTM (random/trainable embeddings)
  8. CNN (random/trainable embeddings)
  9. Multi-channel Word2Vec + CNN
 10. Word2Vec + LSTM
 11. GloVe + LSTM

Important methodological rules
------------------------------
* If results/splits/frd_split.csv already exists, it is REUSED verbatim.
  This is the safest way to guarantee exact sample-level comparability with
  the Transformer experiments.
* If it does not exist, the split is generated using the SAME duplicate-safe
  algorithm as FRD_Revision_AllInOne.py (seed=42; 70/10/20; duplicate groups
  assigned atomically; class+category stratification where possible).
* Vocabulary/vectorizers/Word2Vec/POS transformations are fitted from TRAIN
  data only. The held-out test set is never used for model or threshold
  selection.
* The operating threshold is chosen from validation probabilities only, using
  exactly the same objective/tie-break sequence as the Transformer pipeline:
  Accuracy -> F1 -> ROC-AUC -> closeness to 0.5.

Recommended use (same folder as FRD_Revision_AllInOne.py):
----------------------------------------------------------
  .\.venv\Scripts\python.exe FRD_Run_NonTransformer_Study.py --setup
  .\.venv\Scripts\python.exe FRD_Run_NonTransformer_Study.py --dry-run
  .\.venv\Scripts\python.exe FRD_Run_NonTransformer_Study.py

Dataset expected by default:
  fake reviews dataset.csv

Outputs:
  results/non_transformers/seed_42/<model>/...
  results/paper_pack/non_transformer_metrics.csv
  results/paper_pack/non_transformer_metrics_calibrated.csv
  results/paper_pack/non_transformer_confusion_counts.csv
  results/paper_pack/non_transformer_confidence_intervals.csv
  results/paper_pack/non_transformer_progression.csv
  results/paper_pack/combined_model_metrics.csv       (if Transformer pack exists)
  results/paper_pack/cross_paradigm_statistical_tests.csv

The script is resumable. Completed model folders are skipped unless --force is used.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import joblib

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# Constants / model inventory
# -----------------------------------------------------------------------------
MODEL_INVENTORY = {
    "naive_bayes": {
        "paper_name": "Naive Bayes (NB)",
        "paradigm": "Traditional ML",
        "representation": "Bag-of-Words",
    },
    "logistic_regression": {
        "paper_name": "Logistic Regression (LR)",
        "paradigm": "Traditional ML",
        "representation": "Bag-of-Words",
    },
    "tfidf_lr_pos": {
        "paper_name": "TF-IDF + LR (Additional Features via POS Tagging)",
        "paradigm": "Traditional ML",
        "representation": "TF-IDF + POS profile",
    },
    "tfidf_lr_ngram": {
        "paper_name": "TF-IDF + LR with N-gram Features",
        "paradigm": "Traditional ML",
        "representation": "TF-IDF word 1-2 grams",
    },
    "word2vec_rf": {
        "paper_name": "Word2Vec + Random Forest",
        "paradigm": "Traditional ML",
        "representation": "Mean Word2Vec document vector",
    },
    "tfidf_svm": {
        "paper_name": "TF-IDF + SVM",
        "paradigm": "Traditional ML",
        "representation": "TF-IDF word 1-2 grams",
    },
    "lstm": {
        "paper_name": "LSTM",
        "paradigm": "Deep Learning",
        "representation": "Random trainable word embeddings",
    },
    "cnn": {
        "paper_name": "CNN",
        "paradigm": "Deep Learning",
        "representation": "Random trainable word embeddings",
    },
    "multichannel_word2vec_cnn": {
        "paper_name": "Multi-channel Word2Vec + CNN",
        "paradigm": "Deep Learning",
        "representation": "Train-corpus Word2Vec + parallel CNN kernels",
    },
    "word2vec_lstm": {
        "paper_name": "Word2Vec + LSTM",
        "paradigm": "Deep Learning",
        "representation": "Train-corpus Word2Vec initialization",
    },
    "glove_lstm": {
        "paper_name": "GloVe + LSTM",
        "paradigm": "Deep Learning",
        "representation": "Pretrained GloVe initialization",
    },
}

DEFAULT_MODELS = list(MODEL_INVENTORY.keys())
POS_GROUPS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ", "MODAL", "NUM", "OTHER"]
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.I)


# -----------------------------------------------------------------------------
# Reproducibility / setup
# -----------------------------------------------------------------------------
def setup_extra_dependencies() -> int:
    """Install only the extra packages not already present in the Transformer env."""
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "gensim==4.3.3", "nltk>=3.8,<4"]
    print("[CMD]", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def set_global_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Exact FRD split logic copied from the current Transformer pipeline
# -----------------------------------------------------------------------------
def normalize_for_grouping(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def make_duplicate_safe_split(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    category_col: Optional[str],
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.10,
    test_fraction: float = 0.20,
) -> pd.DataFrame:
    x = df.copy().reset_index(drop=False).rename(columns={"index": "source_index"})
    x["_norm"] = x[text_col].astype(str).map(normalize_for_grouping)

    conflict = x.groupby("_norm")[label_col].nunique()
    bad = conflict[conflict > 1]
    if len(bad):
        raise ValueError(f"Found {len(bad)} normalized duplicate groups with conflicting labels")

    x["_group"] = pd.factorize(x["_norm"], sort=True)[0]
    rows = []
    for gid, g in x.groupby("_group", sort=True):
        cat = str(g[category_col].mode().iloc[0]) if category_col and category_col in g.columns else "NA"
        rows.append({"_group": gid, "label": int(g[label_col].iloc[0]), "category": cat, "n": len(g)})
    groups = pd.DataFrame(rows)
    groups["_stratum"] = groups["label"].astype(str) + "|" + groups["category"].astype(str)
    vc = groups["_stratum"].value_counts()
    groups.loc[groups["_stratum"].map(vc) < 3, "_stratum"] = groups["label"].astype(str)

    try:
        dev_g, test_g = train_test_split(
            groups, test_size=test_fraction, random_state=seed, stratify=groups["_stratum"]
        )
    except ValueError:
        dev_g, test_g = train_test_split(groups, test_size=test_fraction, random_state=seed, stratify=groups["label"])

    relative_val = val_fraction / (train_fraction + val_fraction)
    try:
        train_g, val_g = train_test_split(
            dev_g, test_size=relative_val, random_state=seed, stratify=dev_g["_stratum"]
        )
    except ValueError:
        train_g, val_g = train_test_split(dev_g, test_size=relative_val, random_state=seed, stratify=dev_g["label"])

    mapping = {
        **{int(g): "train" for g in train_g["_group"]},
        **{int(g): "validation" for g in val_g["_group"]},
        **{int(g): "test" for g in test_g["_group"]},
    }
    x["split"] = x["_group"].map(mapping)
    if x["split"].isna().any():
        raise RuntimeError("Split assignment contains missing values")

    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if not set(x.loc[x.split == a, "_norm"]).isdisjoint(set(x.loc[x.split == b, "_norm"])):
            raise RuntimeError(f"Normalized duplicate leakage detected between {a} and {b}")

    return x.drop(columns=["_norm", "_group"])


def load_or_create_frd_split(csv_path: Path, results_dir: Path, split_seed: int = 42) -> Tuple[pd.DataFrame, dict]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"FRD CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    required = {"text_", "label"}
    if not required.issubset(raw.columns):
        raise ValueError(f"FRD missing columns: {required - set(raw.columns)}")

    raw = raw.copy()
    raw["label_num"] = raw["label"].map({"CG": 1, "OR": 0}) if raw["label"].dtype == object else raw["label"].astype(int)
    if raw["label_num"].isna().any():
        raise ValueError("Unrecognized FRD labels")

    split_dir = Path(results_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "frd_split.csv"
    integrity_path = split_dir / "dataset_integrity_report.json"

    if split_path.exists():
        split = pd.read_csv(split_path)
        needed = {"source_index", "split", "label_num", "text_"}
        if not needed.issubset(split.columns):
            raise ValueError(f"Existing split file is incompatible; missing {needed - set(split.columns)}")
        if len(split) != len(raw):
            raise ValueError(f"Existing split has {len(split)} rows but source CSV has {len(raw)}")

        # Hard validation that the persisted split belongs to the supplied source CSV.
        ordered = split.sort_values("source_index").reset_index(drop=True)
        if not np.array_equal(ordered["source_index"].to_numpy(), np.arange(len(raw))):
            raise ValueError("Existing split source_index does not cover the source CSV exactly")
        if not np.array_equal(ordered["label_num"].astype(int).to_numpy(), raw["label_num"].astype(int).to_numpy()):
            raise ValueError("Existing split labels do not match the supplied source CSV")
        split_norm = ordered["text_"].astype(str).map(normalize_for_grouping).to_numpy()
        raw_norm = raw["text_"].astype(str).map(normalize_for_grouping).to_numpy()
        if not np.array_equal(split_norm, raw_norm):
            raise ValueError("Existing split text does not match the supplied source CSV")

        print(f"[SPLIT] Reusing persisted Transformer split: {split_path}")
        report = json.loads(integrity_path.read_text(encoding="utf-8")) if integrity_path.exists() else {}
        return split, report

    category = "category" if "category" in raw.columns else None
    split = make_duplicate_safe_split(raw, "text_", "label_num", category, seed=split_seed)
    split.to_csv(split_path, index=False)

    norm = raw["text_"].astype(str).map(normalize_for_grouping)
    report = {
        "rows": int(len(raw)),
        "unique_normalized_texts": int(norm.nunique()),
        "duplicate_groups": int((norm.value_counts() > 1).sum()),
        "split_counts": {k: int(v) for k, v in split["split"].value_counts().to_dict().items()},
        "split_sha256": sha256_file(split_path),
        "source_sha256": sha256_file(csv_path),
        "cross_split_normalized_overlap": 0,
    }
    integrity_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SPLIT] Created exact duplicate-safe 70/10/20 split: {split_path}")
    return split, report


# -----------------------------------------------------------------------------
# Metrics - mirrors current Transformer implementation
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "fpr": float(fp / (fp + tn) if fp + tn else 0.0),
        "fnr": float(fn / (fn + tp) if fn + tp else 0.0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def choose_threshold(y_true, y_prob, metric: str = "accuracy", grid=None) -> float:
    """Exact-equivalent threshold rule used by the Transformer pipeline.

    ROC-AUC is threshold-independent, so recomputing it for every candidate would
    only waste time. We therefore compute the threshold-dependent confusion counts
    directly while preserving the same ordering: Accuracy -> F1 -> ROC-AUC
    (constant across candidates) -> proximity to 0.5.
    """
    if metric != "accuracy":
        raise ValueError("This study intentionally mirrors the current FRD pipeline, whose threshold objective is accuracy")
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if grid is None:
        uniq = np.unique(p)
        mids = (uniq[:-1] + uniq[1:]) / 2 if len(uniq) > 1 else np.array([0.5])
        grid = np.unique(np.clip(np.concatenate(([0.05, 0.5, 0.95], mids)), 0.01, 0.99))
    best = None
    n = len(y)
    for t in grid:
        pred = p >= float(t)
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum((~pred) & (y == 1)))
        tn = n - tp - fp - fn
        acc = (tp + tn) / n if n else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        score = (acc, f1, -abs(float(t) - 0.5))
        if best is None or score > best[0]:
            best = (score, float(t))
    return best[1]


def bootstrap_ci(y_true, y_prob, metric: str, threshold: float, n_boot: int, seed: int = 314159, alpha: float = 0.05) -> dict:
    if n_boot <= 0:
        return {"low": np.nan, "high": np.nan, "n": 0}
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(compute_metrics(y[idx], p[idx], threshold)[metric])
        except ValueError:
            continue
    arr = np.asarray(vals, dtype=float)
    return {
        "low": float(np.quantile(arr, alpha / 2)) if len(arr) else np.nan,
        "high": float(np.quantile(arr, 1 - alpha / 2)) if len(arr) else np.nan,
        "n": int(len(arr)),
    }


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    y = np.asarray(y_true)
    a = np.asarray(pred_a) == y
    b = np.asarray(pred_b) == y
    b_only = int(np.sum(a & ~b))
    c_only = int(np.sum(~a & b))
    n = b_only + c_only
    if n < 25:
        p = stats.binomtest(min(b_only, c_only), n, 0.5).pvalue if n else 1.0
        chi2 = np.nan
        method = "exact-binomial"
    else:
        chi2 = (abs(b_only - c_only) - 1) ** 2 / n
        p = float(stats.chi2.sf(chi2, 1))
        method = "continuity-corrected"
    return {"b": b_only, "c": c_only, "chi2": chi2, "p": float(p), "method": method}


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, float)
    if len(p) == 0:
        return []
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


# -----------------------------------------------------------------------------
# Text preparation
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    s = URL_RE.sub(" ", str(text))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def basic_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(clean_text(text))


def _pos_group(tag: str) -> str:
    if tag.startswith("NN"):
        return "NOUN"
    if tag.startswith("VB"):
        return "VERB"
    if tag.startswith("JJ"):
        return "ADJ"
    if tag.startswith("RB"):
        return "ADV"
    if tag.startswith("PRP") or tag.startswith("WP"):
        return "PRON"
    if tag in {"DT", "PDT", "WDT"}:
        return "DET"
    if tag in {"IN", "TO"}:
        return "ADP"
    if tag in {"CC"}:
        return "CONJ"
    if tag == "MD":
        return "MODAL"
    if tag == "CD":
        return "NUM"
    return "OTHER"


def ensure_nltk_tagger() -> None:
    import nltk
    try:
        nltk.pos_tag(["test"])
        return
    except LookupError:
        pass
    for resource in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"]:
        try:
            nltk.download(resource, quiet=False)
            nltk.pos_tag(["test"])
            return
        except LookupError:
            continue
    raise RuntimeError("NLTK POS tagger could not be installed/downloaded")


def pos_profile_matrix(texts: Sequence[str], cache_path: Optional[Path] = None) -> np.ndarray:
    if cache_path and cache_path.exists():
        arr = np.load(cache_path)
        if len(arr) == len(texts):
            return arr
    ensure_nltk_tagger()
    import nltk

    tokenized = [basic_tokenize(x) for x in texts]
    tagged = nltk.pos_tag_sents(tokenized)
    out = np.zeros((len(tagged), len(POS_GROUPS) + 2), dtype=np.float32)
    gidx = {g: i for i, g in enumerate(POS_GROUPS)}
    for i, sent in enumerate(tagged):
        n = max(1, len(sent))
        for tok, tag in sent:
            out[i, gidx[_pos_group(tag)]] += 1.0
        out[i, : len(POS_GROUPS)] /= float(n)
        out[i, len(POS_GROUPS)] = float(len(sent)) / 100.0
        out[i, len(POS_GROUPS) + 1] = (sum(len(tok) for tok, _ in sent) / n) / 10.0
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)
    return out


# -----------------------------------------------------------------------------
# Prediction/result persistence
# -----------------------------------------------------------------------------
def save_predictions(path: Path, frame: pd.DataFrame, y_prob, threshold: float, model_key: str, seed: int) -> pd.DataFrame:
    p = np.asarray(y_prob, dtype=float)
    y = frame["label_num"].to_numpy(dtype=int)
    pred = (p >= threshold).astype(int)
    out = pd.DataFrame(
        {
            "sample_id": frame["source_index"].to_numpy() if "source_index" in frame else np.arange(len(frame)),
            "y_true": y,
            "y_prob_cg": p,
            "y_pred": pred,
            "threshold": float(threshold),
            "model": model_key,
            "seed": int(seed),
        }
    )
    if "category" in frame.columns:
        out["category"] = frame["category"].astype(str).to_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(path)
    return out


def write_metrics_bundle(
    out_dir: Path,
    model_key: str,
    seed: int,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    val_prob,
    test_prob,
    resource: dict,
    bootstrap_n: int,
    bootstrap_seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = choose_threshold(val_frame["label_num"].to_numpy(), val_prob)
    payload = {
        "model": model_key,
        "paper_name": MODEL_INVENTORY[model_key]["paper_name"],
        "paradigm": MODEL_INVENTORY[model_key]["paradigm"],
        "representation": MODEL_INVENTORY[model_key]["representation"],
        "seed": int(seed),
        "validation_default": compute_metrics(val_frame["label_num"], val_prob, 0.5),
        "selected_threshold": float(threshold),
        "test_default": compute_metrics(test_frame["label_num"], test_prob, 0.5),
        "test_calibrated": compute_metrics(test_frame["label_num"], test_prob, threshold),
        "resource": resource,
    }
    cis = {}
    for metric in ["accuracy", "f1", "roc_auc"]:
        cis[metric] = bootstrap_ci(
            test_frame["label_num"].to_numpy(), test_prob, metric, threshold, n_boot=bootstrap_n, seed=bootstrap_seed
        )
    payload["bootstrap_95ci"] = cis

    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    save_predictions(out_dir / "validation_predictions.csv", val_frame, val_prob, 0.5, model_key, seed)
    save_predictions(out_dir / "test_predictions.csv", test_frame, test_prob, threshold, model_key, seed)
    return payload


def completed_result(out_dir: Path) -> Optional[dict]:
    m = out_dir / "metrics.json"
    vp = out_dir / "validation_predictions.csv"
    tp = out_dir / "test_predictions.csv"
    if m.exists() and vp.exists() and tp.exists():
        try:
            return json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# Gensim / Word2Vec helpers
# -----------------------------------------------------------------------------
def _import_gensim():
    # gensim 4.3.x expects scipy.linalg.triu on some SciPy releases.
    import scipy.linalg as sl
    if not hasattr(sl, "triu"):
        sl.triu = np.triu
    import gensim
    from gensim.models import Word2Vec
    return gensim, Word2Vec


def train_or_load_word2vec(train_tokens: List[List[str]], cache_dir: Path, seed: int, vector_size: int = 100):
    _, Word2Vec = _import_gensim()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"word2vec_train_seed{seed}_{vector_size}d.model"
    if path.exists():
        print(f"[CACHE] Loading Word2Vec: {path}")
        return Word2Vec.load(str(path))
    print("[W2V] Training Word2Vec on TRAIN partition only...")
    model = Word2Vec(
        sentences=train_tokens,
        vector_size=vector_size,
        window=5,
        min_count=2,
        workers=1,  # deterministic
        sg=1,
        negative=10,
        epochs=10,
        seed=seed,
    )
    model.save(str(path))
    return model


def mean_w2v_vectors(tokenized: Sequence[Sequence[str]], w2v, dim: int) -> np.ndarray:
    out = np.zeros((len(tokenized), dim), dtype=np.float32)
    for i, toks in enumerate(tokenized):
        vecs = [w2v.wv[t] for t in toks if t in w2v.wv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


# -----------------------------------------------------------------------------
# Vocabulary / embeddings for neural models
# -----------------------------------------------------------------------------
def build_vocab(train_tokens: Sequence[Sequence[str]], max_vocab: int = 50000, min_freq: int = 2) -> Tuple[Dict[str, int], List[str]]:
    counter = Counter(tok for sent in train_tokens for tok in sent)
    words = [w for w, c in counter.most_common() if c >= min_freq][: max(0, max_vocab - 2)]
    itos = ["<PAD>", "<UNK>"] + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode_sequences(tokenized: Sequence[Sequence[str]], stoi: Dict[str, int], max_length: int) -> Tuple[np.ndarray, np.ndarray]:
    seqs = np.zeros((len(tokenized), max_length), dtype=np.int32)
    lens = np.ones(len(tokenized), dtype=np.int32)
    unk = stoi["<UNK>"]
    for i, toks in enumerate(tokenized):
        ids = [stoi.get(t, unk) for t in toks[:max_length]]
        if not ids:
            ids = [unk]
        lens[i] = len(ids)
        seqs[i, : len(ids)] = ids
    return seqs, lens


def embedding_matrix_from_w2v(itos: Sequence[str], w2v, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(0.0, 0.05, size=(len(itos), dim)).astype(np.float32)
    mat[0] = 0.0
    for i, w in enumerate(itos[2:], start=2):
        if w in w2v.wv:
            mat[i] = w2v.wv[w]
    return mat


def load_glove_for_vocab(
    itos: Sequence[str],
    seed: int,
    glove_path: Optional[Path] = None,
    glove_name: str = "glove-wiki-gigaword-100",
) -> Tuple[np.ndarray, int, dict]:
    """Load only the vectors required by our vocabulary when a local GloVe txt is supplied;
    otherwise use gensim downloader (cached after first download)."""
    rng = np.random.default_rng(seed)
    target = set(itos[2:])

    if glove_path:
        glove_path = Path(glove_path)
        if not glove_path.exists():
            raise FileNotFoundError(glove_path)
        print(f"[GLOVE] Loading local vectors from {glove_path}")
        found = {}
        dim = None
        with glove_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < 3:
                    continue
                word = parts[0]
                if word not in target:
                    continue
                vec = np.asarray(parts[1:], dtype=np.float32)
                if dim is None:
                    dim = len(vec)
                if len(vec) == dim:
                    found[word] = vec
        if dim is None:
            raise RuntimeError("No matching GloVe vectors were found for the vocabulary")
        mat = rng.normal(0.0, 0.05, size=(len(itos), dim)).astype(np.float32)
        mat[0] = 0.0
        for i, w in enumerate(itos[2:], start=2):
            if w in found:
                mat[i] = found[w]
        meta = {"source": str(glove_path), "dim": dim, "covered": len(found), "vocab": len(target)}
        return mat, dim, meta

    print(f"[GLOVE] Loading/downloading gensim vector set: {glove_name}")
    gensim, _ = _import_gensim()
    import gensim.downloader as api
    kv = api.load(glove_name)
    dim = int(kv.vector_size)
    mat = rng.normal(0.0, 0.05, size=(len(itos), dim)).astype(np.float32)
    mat[0] = 0.0
    covered = 0
    for i, w in enumerate(itos[2:], start=2):
        if w in kv:
            mat[i] = kv[w]
            covered += 1
    meta = {"source": f"gensim:{glove_name}", "dim": dim, "covered": covered, "vocab": len(target)}
    del kv
    gc.collect()
    return mat, dim, meta


# -----------------------------------------------------------------------------
# PyTorch models
# -----------------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, seqs: np.ndarray, lengths: np.ndarray, labels: np.ndarray):
        self.seqs = seqs
        self.lengths = lengths
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.seqs[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        embedding_matrix: Optional[np.ndarray] = None,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(self.dropout(h[-1])).squeeze(1)


class CNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        channels: int = 128,
        kernel_size: int = 5,
        embedding_matrix: Optional[np.ndarray] = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.conv = nn.Conv1d(embedding_dim, channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x, lengths=None):
        z = self.embedding(x).transpose(1, 2)
        z = torch.relu(self.conv(z))
        z = torch.max(z, dim=2).values
        return self.fc(self.dropout(z)).squeeze(1)


class MultiChannelCNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        embedding_matrix: np.ndarray,
        kernels: Sequence[int] = (3, 4, 5),
        channels: int = 100,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, kernel_size=k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernels), 1)

    def forward(self, x, lengths=None):
        z = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(z))
            pooled.append(torch.max(h, dim=2).values)
        z = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(z)).squeeze(1)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters_total": int(total), "parameters_trainable": int(trainable)}


def predict_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for x, lengths, _ in loader:
            x = x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            logits = model(x, lengths)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs)


def train_torch_binary_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    y_val: np.ndarray,
    seed: int,
    device: torch.device,
    max_epochs: int = 10,
    patience: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    set_global_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_f1 = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for x, lengths, y in train_loader:
            x = x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_prob = predict_torch(model, val_loader, device)
        vm = compute_metrics(y_val, val_prob, 0.5)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_f1": vm["f1"], "val_accuracy": vm["accuracy"]})
        print(f"    epoch={epoch:02d} loss={np.mean(losses):.5f} val_acc={vm['accuracy']:.5f} val_f1={vm['f1']:.5f}")

        if vm["f1"] > best_f1 + 1e-12:
            best_f1 = vm["f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    train_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("No valid neural checkpoint was produced")
    model.load_state_dict(best_state)

    pred_start = time.perf_counter()
    val_prob = predict_torch(model, val_loader, device)
    test_prob = predict_torch(model, test_loader, device)
    pred_seconds = time.perf_counter() - pred_start
    n_pred = len(val_prob) + len(test_prob)

    resource = {
        **count_parameters(model),
        "framework": "PyTorch",
        "optimizer": "AdamW",
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "gradient_clip": 1.0,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "best_checkpoint_criterion": "validation_f1_at_0.5",
        "best_epoch": best_epoch,
        "best_validation_f1": float(best_f1),
        "train_seconds": float(train_seconds),
        "inference_seconds": float(pred_seconds),
        "inference_reviews_per_second": float(n_pred / pred_seconds) if pred_seconds else 0.0,
        "device": str(device),
        "peak_vram_gb": float(torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0,
    }
    return val_prob, test_prob, resource, {"history": history, "state_dict": model.state_dict()}


# -----------------------------------------------------------------------------
# Sklearn model runners
# -----------------------------------------------------------------------------
def _predict_prob(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)
    if hasattr(estimator, "decision_function"):
        return expit(np.asarray(estimator.decision_function(X), dtype=float))
    return np.asarray(estimator.predict(X), dtype=float)


def run_bow_model(model_key: str, train, val, test, out_dir: Path, seed: int, bootstrap_n: int, bootstrap_seed: int):
    vectorizer = CountVectorizer(
        preprocessor=clean_text,
        token_pattern=r"(?u)\b\w[\w'’]+\b",
        lowercase=False,
        min_df=2,
        max_features=80000,
        ngram_range=(1, 1),
    )
    start = time.perf_counter()
    Xtr = vectorizer.fit_transform(train["text_"].astype(str))
    Xv = vectorizer.transform(val["text_"].astype(str))
    Xt = vectorizer.transform(test["text_"].astype(str))

    if model_key == "naive_bayes":
        clf = MultinomialNB(alpha=0.1)
    else:
        clf = LogisticRegression(C=2.0, max_iter=2500, solver="liblinear", random_state=seed)
    clf.fit(Xtr, train["label_num"].to_numpy())
    train_seconds = time.perf_counter() - start
    ps = time.perf_counter()
    vp = _predict_prob(clf, Xv)
    tp = _predict_prob(clf, Xt)
    pred_seconds = time.perf_counter() - ps
    resource = {
        "framework": "scikit-learn",
        "vectorizer": "CountVectorizer unigram",
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "train_seconds": float(train_seconds),
        "inference_seconds": float(pred_seconds),
        "inference_reviews_per_second": float((len(val) + len(test)) / pred_seconds) if pred_seconds else 0.0,
    }
    joblib.dump({"vectorizer": vectorizer, "classifier": clf}, out_dir / "model.joblib")
    return write_metrics_bundle(out_dir, model_key, seed, val, test, vp, tp, resource, bootstrap_n, bootstrap_seed)


def run_tfidf_lr_ngram(train, val, test, out_dir: Path, seed: int, bootstrap_n: int, bootstrap_seed: int):
    model_key = "tfidf_lr_ngram"
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        token_pattern=r"(?u)\b\w[\w'’]+\b",
        lowercase=False,
        min_df=2,
        max_df=0.995,
        max_features=120000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    start = time.perf_counter()
    Xtr = vectorizer.fit_transform(train["text_"].astype(str))
    Xv = vectorizer.transform(val["text_"].astype(str))
    Xt = vectorizer.transform(test["text_"].astype(str))
    clf = LogisticRegression(C=2.0, max_iter=2500, solver="liblinear", random_state=seed)
    clf.fit(Xtr, train["label_num"].to_numpy())
    train_seconds = time.perf_counter() - start
    ps = time.perf_counter()
    vp = clf.predict_proba(Xv)[:, 1]
    tp = clf.predict_proba(Xt)[:, 1]
    pred_seconds = time.perf_counter() - ps
    resource = {
        "framework": "scikit-learn",
        "vectorizer": "TF-IDF word 1-2 grams",
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "train_seconds": float(train_seconds),
        "inference_seconds": float(pred_seconds),
        "inference_reviews_per_second": float((len(val) + len(test)) / pred_seconds) if pred_seconds else 0.0,
    }
    joblib.dump({"vectorizer": vectorizer, "classifier": clf}, out_dir / "model.joblib")
    return write_metrics_bundle(out_dir, model_key, seed, val, test, vp, tp, resource, bootstrap_n, bootstrap_seed)


def run_tfidf_lr_pos(frame, train, val, test, out_dir: Path, results_dir: Path, seed: int, bootstrap_n: int, bootstrap_seed: int):
    model_key = "tfidf_lr_pos"
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        token_pattern=r"(?u)\b\w[\w'’]+\b",
        lowercase=False,
        min_df=2,
        max_df=0.995,
        max_features=100000,
        ngram_range=(1, 1),
        sublinear_tf=True,
    )
    cache = results_dir / "non_transformers" / "cache" / "pos_profile_all.npy"
    print("[POS] Building/loading POS feature profile...")
    pos_all = pos_profile_matrix(frame["text_"].astype(str).tolist(), cache)
    pos_by_source = {int(src): pos_all[i] for i, src in enumerate(frame["source_index"].to_numpy())}
    Ptr = np.vstack([pos_by_source[int(i)] for i in train["source_index"]])
    Pv = np.vstack([pos_by_source[int(i)] for i in val["source_index"]])
    Pt = np.vstack([pos_by_source[int(i)] for i in test["source_index"]])

    start = time.perf_counter()
    Ttr = vectorizer.fit_transform(train["text_"].astype(str))
    Tv = vectorizer.transform(val["text_"].astype(str))
    Tt = vectorizer.transform(test["text_"].astype(str))
    Xtr = sparse.hstack([Ttr, sparse.csr_matrix(Ptr)], format="csr")
    Xv = sparse.hstack([Tv, sparse.csr_matrix(Pv)], format="csr")
    Xt = sparse.hstack([Tt, sparse.csr_matrix(Pt)], format="csr")
    clf = LogisticRegression(C=2.0, max_iter=2500, solver="liblinear", random_state=seed)
    clf.fit(Xtr, train["label_num"].to_numpy())
    train_seconds = time.perf_counter() - start
    ps = time.perf_counter()
    vp = clf.predict_proba(Xv)[:, 1]
    tp = clf.predict_proba(Xt)[:, 1]
    pred_seconds = time.perf_counter() - ps
    resource = {
        "framework": "scikit-learn + NLTK",
        "vectorizer": "TF-IDF unigram + normalized POS profile",
        "pos_features": POS_GROUPS + ["token_count_scaled", "mean_token_length_scaled"],
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "train_seconds": float(train_seconds),
        "inference_seconds": float(pred_seconds),
        "inference_reviews_per_second": float((len(val) + len(test)) / pred_seconds) if pred_seconds else 0.0,
    }
    joblib.dump({"vectorizer": vectorizer, "classifier": clf, "pos_groups": POS_GROUPS}, out_dir / "model.joblib")
    return write_metrics_bundle(out_dir, model_key, seed, val, test, vp, tp, resource, bootstrap_n, bootstrap_seed)


def run_tfidf_svm(train, val, test, out_dir: Path, seed: int, bootstrap_n: int, bootstrap_seed: int):
    model_key = "tfidf_svm"
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        token_pattern=r"(?u)\b\w[\w'’]+\b",
        lowercase=False,
        min_df=2,
        max_df=0.995,
        max_features=120000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    start = time.perf_counter()
    Xtr = vectorizer.fit_transform(train["text_"].astype(str))
    Xv = vectorizer.transform(val["text_"].astype(str))
    Xt = vectorizer.transform(test["text_"].astype(str))
    base = LinearSVC(C=1.0, random_state=seed)
    # Calibration uses only internal CV folds of the TRAIN partition.
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=-1)
    clf.fit(Xtr, train["label_num"].to_numpy())
    train_seconds = time.perf_counter() - start
    ps = time.perf_counter()
    vp = clf.predict_proba(Xv)[:, 1]
    tp = clf.predict_proba(Xt)[:, 1]
    pred_seconds = time.perf_counter() - ps
    resource = {
        "framework": "scikit-learn",
        "vectorizer": "TF-IDF word 1-2 grams",
        "classifier": "LinearSVC + train-only 3-fold sigmoid calibration",
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "train_seconds": float(train_seconds),
        "inference_seconds": float(pred_seconds),
        "inference_reviews_per_second": float((len(val) + len(test)) / pred_seconds) if pred_seconds else 0.0,
    }
    joblib.dump({"vectorizer": vectorizer, "classifier": clf}, out_dir / "model.joblib")
    return write_metrics_bundle(out_dir, model_key, seed, val, test, vp, tp, resource, bootstrap_n, bootstrap_seed)


def run_word2vec_rf(train_tokens, val_tokens, test_tokens, train, val, test, out_dir: Path, w2v, seed: int, bootstrap_n: int, bootstrap_seed: int):
    model_key = "word2vec_rf"
    dim = int(w2v.vector_size)
    start = time.perf_counter()
    Xtr = mean_w2v_vectors(train_tokens, w2v, dim)
    Xv = mean_w2v_vectors(val_tokens, w2v, dim)
    Xt = mean_w2v_vectors(test_tokens, w2v, dim)
    clf = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(Xtr, train["label_num"].to_numpy())
    train_seconds = time.perf_counter() - start
    ps = time.perf_counter()
    vp = clf.predict_proba(Xv)[:, 1]
    tp = clf.predict_proba(Xt)[:, 1]
    pred_seconds = time.perf_counter() - ps
    resource = {
        "framework": "gensim + scikit-learn",
        "word2vec_dim": dim,
        "word2vec_training_partition": "train_only",
        "random_forest_trees": 400,
        "train_seconds": float(train_seconds),
        "inference_seconds": float(pred_seconds),
        "inference_reviews_per_second": float((len(val) + len(test)) / pred_seconds) if pred_seconds else 0.0,
    }
    joblib.dump(clf, out_dir / "random_forest.joblib")
    return write_metrics_bundle(out_dir, model_key, seed, val, test, vp, tp, resource, bootstrap_n, bootstrap_seed)


# -----------------------------------------------------------------------------
# Neural model orchestration
# -----------------------------------------------------------------------------
def make_sequence_loaders(
    train_tokens,
    val_tokens,
    test_tokens,
    train,
    val,
    test,
    cache_dir: Path,
    max_length: int,
    max_vocab: int,
    batch_size: int,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = cache_dir / f"vocab_{max_vocab}.json"
    seq_path = cache_dir / f"encoded_sequences_v{max_vocab}_l{max_length}.npz"

    if vocab_path.exists():
        itos = json.loads(vocab_path.read_text(encoding="utf-8"))
        stoi = {w: i for i, w in enumerate(itos)}
    else:
        stoi, itos = build_vocab(train_tokens, max_vocab=max_vocab, min_freq=2)
        vocab_path.write_text(json.dumps(itos, ensure_ascii=False), encoding="utf-8")

    if seq_path.exists():
        z = np.load(seq_path)
        tr_seq, tr_len = z["tr_seq"], z["tr_len"]
        va_seq, va_len = z["va_seq"], z["va_len"]
        te_seq, te_len = z["te_seq"], z["te_len"]
    else:
        tr_seq, tr_len = encode_sequences(train_tokens, stoi, max_length)
        va_seq, va_len = encode_sequences(val_tokens, stoi, max_length)
        te_seq, te_len = encode_sequences(test_tokens, stoi, max_length)
        np.savez_compressed(seq_path, tr_seq=tr_seq, tr_len=tr_len, va_seq=va_seq, va_len=va_len, te_seq=te_seq, te_len=te_len)

    tr_ds = SequenceDataset(tr_seq, tr_len, train["label_num"].to_numpy())
    va_ds = SequenceDataset(va_seq, va_len, val["label_num"].to_numpy())
    te_ds = SequenceDataset(te_seq, te_len, test["label_num"].to_numpy())
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    va_loader = DataLoader(va_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    te_loader = DataLoader(te_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    return stoi, itos, tr_loader, va_loader, te_loader


def run_neural_model(
    model_key: str,
    train,
    val,
    test,
    loaders,
    itos,
    w2v,
    out_dir: Path,
    seed: int,
    device: torch.device,
    bootstrap_n: int,
    bootstrap_seed: int,
    glove_path: Optional[Path],
    glove_name: str,
    max_epochs: int,
):
    _, _, tr_loader, va_loader, te_loader = loaders
    vocab_size = len(itos)
    embedding_meta = {}

    set_global_seed(seed)
    if model_key == "lstm":
        model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128)
        embedding_meta = {"embedding": "random_trainable", "dimension": 100}
    elif model_key == "cnn":
        model = CNNClassifier(vocab_size=vocab_size, embedding_dim=100, channels=128, kernel_size=5)
        embedding_meta = {"embedding": "random_trainable", "dimension": 100, "kernel_size": 5}
    elif model_key == "word2vec_lstm":
        mat = embedding_matrix_from_w2v(itos, w2v, int(w2v.vector_size), seed)
        model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=int(w2v.vector_size), hidden_dim=128, embedding_matrix=mat)
        embedding_meta = {"embedding": "Word2Vec_train_only_trainable", "dimension": int(w2v.vector_size)}
    elif model_key == "multichannel_word2vec_cnn":
        mat = embedding_matrix_from_w2v(itos, w2v, int(w2v.vector_size), seed)
        model = MultiChannelCNNClassifier(vocab_size=vocab_size, embedding_dim=int(w2v.vector_size), embedding_matrix=mat, kernels=(3, 4, 5), channels=100)
        embedding_meta = {"embedding": "Word2Vec_train_only_trainable", "dimension": int(w2v.vector_size), "kernels": [3, 4, 5]}
    elif model_key == "glove_lstm":
        mat, dim, meta = load_glove_for_vocab(itos, seed, glove_path=glove_path, glove_name=glove_name)
        model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=dim, hidden_dim=128, embedding_matrix=mat)
        embedding_meta = {"embedding": "GloVe_pretrained_trainable", **meta}
    else:
        raise KeyError(model_key)

    vp, tp, resource, art = train_torch_binary_classifier(
        model,
        tr_loader,
        va_loader,
        te_loader,
        val["label_num"].to_numpy(),
        seed=seed,
        device=device,
        max_epochs=max_epochs,
        patience=2,
        lr=1e-3,
        weight_decay=1e-4,
    )
    resource.update(embedding_meta)
    torch.save({"model_key": model_key, "state_dict": art["state_dict"], "itos": itos, "resource": resource}, out_dir / "model.pt")
    (out_dir / "training_history.json").write_text(json.dumps(art["history"], indent=2), encoding="utf-8")
    return write_metrics_bundle(out_dir, model_key, seed, val, test, vp, tp, resource, bootstrap_n, bootstrap_seed)


# -----------------------------------------------------------------------------
# Reporting / cross-paradigm significance
# -----------------------------------------------------------------------------
def metric_rows_from_payload(payload: dict) -> List[dict]:
    rows = []
    for mode_key, mode_name in [("test_default", "default_0.5"), ("test_calibrated", "validation_calibrated")]:
        m = payload[mode_key]
        rows.append(
            {
                "protocol": "frd",
                "seed": payload["seed"],
                "model": payload["model"],
                "paper_name": payload.get("paper_name", payload["model"]),
                "paradigm": payload.get("paradigm"),
                "representation": payload.get("representation"),
                "decision_rule": mode_name,
                "selected_threshold": payload.get("selected_threshold"),
                **m,
            }
        )
    return rows


def build_nontransformer_reports(results_dir: Path, seed: int, model_payloads: Dict[str, dict]) -> None:
    paper = results_dir / "paper_pack"
    paper.mkdir(parents=True, exist_ok=True)
    rows = []
    ci_rows = []
    conf_rows = []
    inv_rows = []

    for key in DEFAULT_MODELS:
        p = model_payloads.get(key)
        if not p:
            continue
        rows.extend(metric_rows_from_payload(p))
        tm = p["test_calibrated"]
        conf_rows.append(
            {
                "model": key,
                "paper_name": p.get("paper_name", key),
                "paradigm": p.get("paradigm"),
                "threshold": p["selected_threshold"],
                "tn": tm["tn"],
                "fp": tm["fp"],
                "fn": tm["fn"],
                "tp": tm["tp"],
            }
        )
        for metric, ci in p.get("bootstrap_95ci", {}).items():
            ci_rows.append({"model": key, "paper_name": p.get("paper_name", key), "metric": metric, **ci})
        inv_rows.append({"model": key, **MODEL_INVENTORY[key], **p.get("resource", {})})

    df = pd.DataFrame(rows)
    df.to_csv(paper / "non_transformer_metrics.csv", index=False)
    if not df.empty:
        df[df.decision_rule == "validation_calibrated"].to_csv(paper / "non_transformer_metrics_calibrated.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(paper / "non_transformer_confusion_counts.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(paper / "non_transformer_confidence_intervals.csv", index=False)
    pd.DataFrame(inv_rows).to_csv(paper / "non_transformer_model_inventory.csv", index=False)

    # Select paradigm representatives ONLY from validation metrics, never from test results.
    prog = []
    for paradigm in ["Traditional ML", "Deep Learning"]:
        cand = [p for p in model_payloads.values() if p.get("paradigm") == paradigm]
        if not cand:
            continue
        best = max(
            cand,
            key=lambda p: (
                p["validation_default"]["accuracy"],
                p["validation_default"]["f1"],
                p["validation_default"]["roc_auc"],
                p["model"],
            ),
        )
        prog.append(
            {
                "paradigm": paradigm,
                "validation_selected_model": best["paper_name"],
                "model_key": best["model"],
                "selected_threshold": best["selected_threshold"],
                **{k: best["test_calibrated"][k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]},
            }
        )
    prog_df = pd.DataFrame(prog)
    if len(prog_df) >= 2:
        prog_df["accuracy_delta_pp_from_previous"] = prog_df["accuracy"].diff() * 100.0
        prog_df["f1_delta_pp_from_previous"] = prog_df["f1"].diff() * 100.0
    prog_df.to_csv(paper / "non_transformer_progression.csv", index=False)

    # Append to current Transformer paper pack when it exists.
    trans = paper / "model_metrics.csv"
    if trans.exists() and not df.empty:
        tdf = pd.read_csv(trans)
        if "paper_name" not in tdf.columns:
            tdf["paper_name"] = tdf.get("model", "")
        if "paradigm" not in tdf.columns:
            tdf["paradigm"] = "Transformer"
        if "representation" not in tdf.columns:
            tdf["representation"] = "Pretrained contextual representation"
        combined = pd.concat([df, tdf], ignore_index=True, sort=False)
        combined.to_csv(paper / "combined_model_metrics.csv", index=False)

    run_cross_paradigm_tests(results_dir, seed, model_payloads, prog_df)


def discover_best_transformer_bundle(results_dir: Path, seed: int) -> Optional[dict]:
    base = results_dir / "models" / "frd" / f"seed_{seed}"
    candidates = []
    if not base.exists():
        return None
    for sel in base.glob("*/selected_result.json"):
        try:
            meta = json.loads(sel.read_text(encoding="utf-8"))
            metrics = meta
            if "validation_default" not in metrics:
                continue
            result_dir = Path(meta.get("result_dir", sel.parent))
            tp = result_dir / "test_predictions.csv"
            if not tp.exists():
                continue
            candidates.append((metrics, tp))
        except Exception:
            continue
    if not candidates:
        return None
    metrics, tp = max(
        candidates,
        key=lambda it: (
            it[0]["validation_default"]["accuracy"],
            it[0]["validation_default"]["f1"],
            it[0]["validation_default"]["roc_auc"],
            it[0].get("model", ""),
        ),
    )
    return {"name": metrics.get("model", "transformer"), "metrics": metrics, "pred_path": tp}


def discover_best_ensemble_bundle(results_dir: Path, seed: int) -> Optional[dict]:
    base = results_dir / "ensembles" / "frd" / f"seed_{seed}"
    candidates = []
    if not base.exists():
        return None
    for mp in base.glob("*/*/metrics.json"):
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            tp = mp.parent / "test_predictions.csv"
            if "validation_default" in m and tp.exists():
                candidates.append((m, tp))
        except Exception:
            continue
    if not candidates:
        return None
    m, tp = max(
        candidates,
        key=lambda it: (
            it[0]["validation_default"]["accuracy"],
            it[0]["validation_default"]["f1"],
            it[0]["validation_default"]["roc_auc"],
            it[0].get("model", ""),
        ),
    )
    return {"name": m.get("model", "ensemble"), "metrics": m, "pred_path": tp}


def load_nontransformer_pred(results_dir: Path, seed: int, key: str) -> pd.DataFrame:
    return pd.read_csv(results_dir / "non_transformers" / f"seed_{seed}" / key / "test_predictions.csv")


def paired_test_from_frames(name_a: str, a: pd.DataFrame, name_b: str, b: pd.DataFrame) -> Optional[dict]:
    a = a.sort_values("sample_id").reset_index(drop=True)
    b = b.sort_values("sample_id").reset_index(drop=True)
    if not np.array_equal(a["sample_id"].to_numpy(), b["sample_id"].to_numpy()):
        return None
    if not np.array_equal(a["y_true"].to_numpy(), b["y_true"].to_numpy()):
        return None
    mc = mcnemar_test(a["y_true"], a["y_pred"], b["y_pred"])
    return {"model_a": name_a, "model_b": name_b, **mc}


def run_cross_paradigm_tests(results_dir: Path, seed: int, payloads: Dict[str, dict], prog_df: pd.DataFrame) -> None:
    paper = results_dir / "paper_pack"
    comparisons = []
    bundles = []

    if not prog_df.empty:
        for _, row in prog_df.iterrows():
            key = row["model_key"]
            p = results_dir / "non_transformers" / f"seed_{seed}" / key / "test_predictions.csv"
            if p.exists():
                bundles.append((row["paradigm"], row["validation_selected_model"], pd.read_csv(p)))

    tr = discover_best_transformer_bundle(results_dir, seed)
    if tr:
        bundles.append(("Transformer", tr["name"], pd.read_csv(tr["pred_path"])))
    ens = discover_best_ensemble_bundle(results_dir, seed)
    if ens:
        bundles.append(("Ensemble", ens["name"], pd.read_csv(ens["pred_path"])))

    # Planned progression comparisons only: Traditional -> DL -> Transformer -> Ensemble.
    for i in range(len(bundles) - 1):
        pa, na, a = bundles[i]
        pb, nb, b = bundles[i + 1]
        r = paired_test_from_frames(na, a, nb, b)
        if r:
            r["paradigm_a"] = pa
            r["paradigm_b"] = pb
            comparisons.append(r)

    if comparisons:
        adj = holm_adjust([r["p"] for r in comparisons])
        for r, ap in zip(comparisons, adj):
            r["holm_p"] = ap
            r["significant_0_05"] = bool(ap < 0.05)
    pd.DataFrame(comparisons).to_csv(paper / "cross_paradigm_statistical_tests.csv", index=False)


# -----------------------------------------------------------------------------
# Main study runner
# -----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--csv", type=Path, default=here / "fake reviews dataset.csv")
    ap.add_argument("--results-dir", type=Path, default=here / "results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples per metric; 0 disables CIs")
    ap.add_argument("--bootstrap-seed", type=int, default=314159)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-vocab", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS), help="Comma-separated model keys")
    ap.add_argument("--glove-path", type=Path, default=None, help="Optional local GloVe .txt file")
    ap.add_argument("--glove-name", type=str, default="glove-wiki-gigaword-100")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--setup", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.setup:
        return setup_extra_dependencies()

    selected = [x.strip() for x in args.models.split(",") if x.strip()]
    unknown = [x for x in selected if x not in MODEL_INVENTORY]
    if unknown:
        raise SystemExit(f"Unknown model key(s): {unknown}\nAllowed: {DEFAULT_MODELS}")

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)

    frame, integrity = load_or_create_frd_split(args.csv.resolve(), results_dir, split_seed=42)
    train = frame[frame.split == "train"].copy().reset_index(drop=True)
    val = frame[frame.split == "validation"].copy().reset_index(drop=True)
    test = frame[frame.split == "test"].copy().reset_index(drop=True)
    print("FRD split counts:", {"train": len(train), "validation": len(val), "test": len(test)})
    print("Class counts:")
    print(frame.groupby(["split", "label_num"]).size())

    expected = {"train": 28303, "validation": 4043, "test": 8086}
    got = {"train": len(train), "validation": len(val), "test": len(test)}
    if got != expected:
        print(f"[WARNING] Split counts differ from the current paper values. Expected {expected}, got {got}.")
        print("          If Transformer results already exist, ensure both scripts point to the same results/splits/frd_split.csv.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nModels to run:")
    for k in selected:
        print(f"  - {k}: {MODEL_INVENTORY[k]['paper_name']}")
    if args.dry_run:
        print("\n[DRY RUN] No training started.")
        return 0

    # Tokenization is shared by W2V and all neural systems; training text only is used to fit vocabulary/Word2Vec.
    need_tokens = any(k in {"word2vec_rf", "lstm", "cnn", "multichannel_word2vec_cnn", "word2vec_lstm", "glove_lstm"} for k in selected)
    train_tokens = val_tokens = test_tokens = None
    w2v = None
    loaders = None
    itos = None
    cache_dir = results_dir / "non_transformers" / "cache"

    if need_tokens:
        print("[TEXT] Tokenizing train/validation/test for static-embedding and neural models...")
        train_tokens = [basic_tokenize(x) for x in train["text_"].astype(str)]
        val_tokens = [basic_tokenize(x) for x in val["text_"].astype(str)]
        test_tokens = [basic_tokenize(x) for x in test["text_"].astype(str)]

    need_w2v = any(k in {"word2vec_rf", "multichannel_word2vec_cnn", "word2vec_lstm"} for k in selected)
    if need_w2v:
        try:
            w2v = train_or_load_word2vec(train_tokens, cache_dir, args.seed, vector_size=100)
        except ImportError as exc:
            raise SystemExit(f"gensim is required for Word2Vec models: {exc}\nRun: {sys.executable} {Path(__file__).name} --setup")

    need_neural = any(k in {"lstm", "cnn", "multichannel_word2vec_cnn", "word2vec_lstm", "glove_lstm"} for k in selected)
    if need_neural:
        stoi, itos, tr_loader, va_loader, te_loader = make_sequence_loaders(
            train_tokens,
            val_tokens,
            test_tokens,
            train,
            val,
            test,
            cache_dir,
            max_length=args.max_length,
            max_vocab=args.max_vocab,
            batch_size=args.batch_size,
        )
        loaders = (stoi, itos, tr_loader, va_loader, te_loader)
        print(f"[VOCAB] size={len(itos)} | max_length={args.max_length}")

    payloads: Dict[str, dict] = {}
    base = results_dir / "non_transformers" / f"seed_{args.seed}"
    base.mkdir(parents=True, exist_ok=True)

    for idx, key in enumerate(selected, start=1):
        out_dir = base / key
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 88)
        print(f"[{idx}/{len(selected)}] {MODEL_INVENTORY[key]['paper_name']}")
        print("=" * 88)

        existing = completed_result(out_dir)
        if existing and not args.force:
            print("[SKIP] Complete metrics + prediction files already exist.")
            payloads[key] = existing
            continue

        # Remove only this model's partial outputs when force/restart is requested.
        if args.force:
            for p in [out_dir / "metrics.json", out_dir / "validation_predictions.csv", out_dir / "test_predictions.csv"]:
                if p.exists():
                    p.unlink()

        set_global_seed(args.seed)
        try:
            if key in {"naive_bayes", "logistic_regression"}:
                payload = run_bow_model(key, train, val, test, out_dir, args.seed, args.bootstrap, args.bootstrap_seed)
            elif key == "tfidf_lr_ngram":
                payload = run_tfidf_lr_ngram(train, val, test, out_dir, args.seed, args.bootstrap, args.bootstrap_seed)
            elif key == "tfidf_lr_pos":
                payload = run_tfidf_lr_pos(frame, train, val, test, out_dir, results_dir, args.seed, args.bootstrap, args.bootstrap_seed)
            elif key == "tfidf_svm":
                payload = run_tfidf_svm(train, val, test, out_dir, args.seed, args.bootstrap, args.bootstrap_seed)
            elif key == "word2vec_rf":
                payload = run_word2vec_rf(train_tokens, val_tokens, test_tokens, train, val, test, out_dir, w2v, args.seed, args.bootstrap, args.bootstrap_seed)
            elif key in {"lstm", "cnn", "multichannel_word2vec_cnn", "word2vec_lstm", "glove_lstm"}:
                if key in {"multichannel_word2vec_cnn", "word2vec_lstm"} and w2v is None:
                    w2v = train_or_load_word2vec(train_tokens, cache_dir, args.seed, vector_size=100)
                payload = run_neural_model(
                    key,
                    train,
                    val,
                    test,
                    loaders,
                    itos,
                    w2v,
                    out_dir,
                    args.seed,
                    device,
                    args.bootstrap,
                    args.bootstrap_seed,
                    args.glove_path,
                    args.glove_name,
                    args.max_epochs,
                )
            else:
                raise KeyError(key)
            payloads[key] = payload
            m = payload["test_calibrated"]
            print(
                f"[DONE] acc={m['accuracy']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} "
                f"f1={m['f1']:.4f} roc_auc={m['roc_auc']:.4f} mcc={m['mcc']:.4f} threshold={payload['selected_threshold']:.4f}"
            )
        except Exception as exc:
            err = {"model": key, "error": repr(exc), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            (out_dir / "error.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
            print(f"[ERROR] {key}: {exc}", file=sys.stderr)
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Load any previously completed models not selected in this invocation, so reports stay complete.
    for key in DEFAULT_MODELS:
        if key not in payloads:
            p = completed_result(base / key)
            if p:
                payloads[key] = p

    build_nontransformer_reports(results_dir, args.seed, payloads)
    print("\n" + "=" * 88)
    print("STUDY COMPLETE")
    print("Paper-ready outputs:", results_dir / "paper_pack")
    print("Exact split:", results_dir / "splits" / "frd_split.csv")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
