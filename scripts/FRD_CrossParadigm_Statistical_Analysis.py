#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FRD_CrossParadigm_Statistical_Analysis.py
=========================================

Final paired statistical analysis for the FRD study.

Purpose
-------
This script DOES NOT train any model and DOES NOT use the held-out test set
for model selection. It reuses already-saved validation/test artifacts to:

1) Select one Traditional-ML representative from validation metrics only.
2) Select one Deep-Learning representative from validation metrics only.
3) Select one plain Transformer representative from validation metrics only.
4) Reconstruct the validation-selected plain-vs-sentiment representative pool
   (one representative per Transformer family), then select the strongest
   individual representative from validation metrics only.
5) Load the already-selected rank-1 weighted soft-voting ensemble from the
   sentiment-extension manifest WITHOUT reselecting members/weights on test.
6) Perform paired McNemar tests on the same 8,086 FRD test observations.
7) Estimate paired bootstrap effect sizes (B - A) for Accuracy, F1 and ROC-AUC.
8) Recompute the seven matched plain-vs-sentiment ablations with a separate
   Holm correction family.

The progression comparison is a SYSTEM-LEVEL comparison. In particular, the
plain-Transformer -> extended-individual transition does not isolate sentiment
causally if the selected backbone changes. Use the matched sentiment-ablation
outputs for claims about the incremental effect of sentiment.

Expected project layout
-----------------------
results/
  non_transformers/seed_42/<model>/metrics.json
  non_transformers/seed_42/<model>/validation_predictions.csv
  non_transformers/seed_42/<model>/test_predictions.csv

  models/frd/seed_42/<transformer>/selected_result.json
  models/frd/seed_42/<transformer>/<attempt>/metrics.json
  models/frd/seed_42/<transformer>/<attempt>/validation_predictions.csv
  models/frd/seed_42/<transformer>/<attempt>/test_predictions.csv

  sentiment_extension/seed_42/ensembles/frd/selection_manifest.json

Outputs
-------
results/paper_pack/final_statistics/
  stage_selection_and_metrics.csv
  stage_confidence_intervals.csv
  planned_progression_paired_tests.csv
  planned_progression_bootstrap_effects.csv
  transformer_representative_selection.csv
  sentiment_ablation_paired_tests.csv
  sentiment_ablation_bootstrap_effects.csv
  ensemble_vs_selected_individual.csv
  selection_audit.json
  final_statistics_summary.md

Primary conventions
-------------------
* Positive class: CG / machine-generated = 1.
* Model/stage selection: VALIDATION ONLY.
* Test operating thresholds: the thresholds already selected on validation.
* Test labels are used only for final metrics/statistical inference.
* Bootstrap: paired resampling of identical test observations.
* Multiple-comparison correction:
    - planned progression family: Holm across adjacent stage comparisons;
    - sentiment ablation family: Holm across seven matched backbone pairs.

No AiGen/OpSpam data are required by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

RUNNER_VERSION = "1.0.0"

# -----------------------------------------------------------------------------
# Study inventory
# -----------------------------------------------------------------------------

TRADITIONAL_KEYS = [
    "naive_bayes",
    "logistic_regression",
    "tfidf_lr_pos",
    "tfidf_lr_ngram",
    "word2vec_rf",
    "tfidf_svm",
]

DEEP_KEYS = [
    "lstm",
    "cnn",
    "multichannel_word2vec_cnn",
    "word2vec_lstm",
    "glove_lstm",
]

PLAIN_TRANSFORMERS = [
    "bert",
    "roberta",
    "deberta",
    "deberta_v3_base",
    "modernbert_base",
    "electra_base",
    "distilbert",
]

SENTIMENT_TRANSFORMERS = [
    "bert_sentiment_profile",
    "roberta_sentiment_profile",
    "deberta_sentiment_profile",
    "deberta_v3_sentiment_gated",
    "modernbert_sentiment_profile",
    "electra_sentiment_profile",
    "distilbert_sentiment_profile",
]

BACKBONE_PAIRS = {
    "BERT": ("bert", "bert_sentiment_profile"),
    "RoBERTa": ("roberta", "roberta_sentiment_profile"),
    "DeBERTa": ("deberta", "deberta_sentiment_profile"),
    "DeBERTa-v3": ("deberta_v3_base", "deberta_v3_sentiment_gated"),
    "ModernBERT": ("modernbert_base", "modernbert_sentiment_profile"),
    "ELECTRA": ("electra_base", "electra_sentiment_profile"),
    "DistilBERT": ("distilbert", "distilbert_sentiment_profile"),
}

DISPLAY_NAMES = {
    "naive_bayes": "Naive Bayes (NB)",
    "logistic_regression": "Logistic Regression (LR)",
    "tfidf_lr_pos": "TF-IDF + LR (Additional Features via POS Tagging)",
    "tfidf_lr_ngram": "TF-IDF + LR with N-gram Features",
    "word2vec_rf": "Word2Vec + Random Forest",
    "tfidf_svm": "TF-IDF + SVM",
    "lstm": "LSTM",
    "cnn": "CNN",
    "multichannel_word2vec_cnn": "Multi-channel Word2Vec + CNN",
    "word2vec_lstm": "Word2Vec + LSTM",
    "glove_lstm": "GloVe + LSTM",
    "bert": "BERT",
    "bert_sentiment_profile": "BERT + Sentiment Profile",
    "roberta": "RoBERTa",
    "roberta_sentiment_profile": "RoBERTa + Sentiment Profile",
    "deberta": "DeBERTa",
    "deberta_sentiment_profile": "DeBERTa + Sentiment Profile",
    "deberta_v3_base": "DeBERTa-v3",
    "deberta_v3_sentiment_gated": "DeBERTa-v3 + Sentiment Profile (Gated)",
    "modernbert_base": "ModernBERT",
    "modernbert_sentiment_profile": "ModernBERT + Sentiment Profile",
    "electra_base": "ELECTRA",
    "electra_sentiment_profile": "ELECTRA + Sentiment Profile",
    "distilbert": "DistilBERT",
    "distilbert_sentiment_profile": "DistilBERT + Sentiment Profile",
}


@dataclass
class Bundle:
    key: str
    display_name: str
    source_type: str
    metrics: dict
    validation: pd.DataFrame
    test: pd.DataFrame
    threshold: float
    metrics_path: Path
    validation_path: Path
    test_path: Path


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _normalise_prediction_frame(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "sample_id" not in df.columns:
        if "source_index" in df.columns:
            df = df.rename(columns={"source_index": "sample_id"})
        else:
            raise ValueError(f"Prediction file has no sample_id/source_index: {path}")
    if "y_true" not in df.columns:
        if "label_num" in df.columns:
            df = df.rename(columns={"label_num": "y_true"})
        else:
            raise ValueError(f"Prediction file has no y_true/label_num: {path}")

    prob_candidates = [
        "y_prob_cg",
        "y_prob_machine_generated",
        "y_prob",
        "probability",
        "prob",
    ]
    prob_col = next((c for c in prob_candidates if c in df.columns), None)
    if prob_col is None:
        raise ValueError(f"Prediction file has no recognized probability column: {path}")
    if prob_col != "y_prob_cg":
        df = df.rename(columns={prob_col: "y_prob_cg"})

    out = df.copy()
    # IDs are normalized to strings only for alignment. This avoids int/string
    # mismatches while preserving exact identity.
    out["sample_id"] = out["sample_id"].astype(str)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="raise").astype(int)
    out["y_prob_cg"] = pd.to_numeric(out["y_prob_cg"], errors="raise").astype(float)

    if out["sample_id"].duplicated().any():
        dup = out.loc[out["sample_id"].duplicated(), "sample_id"].iloc[0]
        raise ValueError(f"Duplicate sample_id={dup} in {path}")
    if not out["y_true"].isin([0, 1]).all():
        raise ValueError(f"Non-binary y_true in {path}")
    if not np.isfinite(out["y_prob_cg"].to_numpy()).all():
        raise ValueError(f"Non-finite probabilities in {path}")
    if ((out["y_prob_cg"] < 0) | (out["y_prob_cg"] > 1)).any():
        raise ValueError(f"Probabilities outside [0,1] in {path}")

    return out[["sample_id", "y_true", "y_prob_cg"]].sort_values("sample_id").reset_index(drop=True)


def _load_prediction(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return _normalise_prediction_frame(pd.read_csv(path), path)


def _threshold_from_payload(payload: dict, test: Optional[pd.DataFrame] = None, pred_path: Optional[Path] = None) -> float:
    value = payload.get("selected_threshold")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    # Fallback: some prediction files carry a constant threshold column.
    if pred_path and pred_path.is_file():
        raw = pd.read_csv(pred_path, usecols=lambda c: c == "threshold")
        if "threshold" in raw.columns:
            vals = pd.to_numeric(raw["threshold"], errors="coerce").dropna().unique()
            if len(vals) == 1:
                return float(vals[0])
    raise ValueError("Could not resolve validation-selected threshold")


def _display(key: str) -> str:
    return DISPLAY_NAMES.get(key, key)


def _validation_tuple(bundle: Bundle) -> Tuple[float, float, float, str]:
    m = bundle.metrics.get("validation_default", {})
    return (
        float(m.get("accuracy", -np.inf)),
        float(m.get("f1", -np.inf)),
        float(m.get("roc_auc", -np.inf)),
        bundle.key,
    )


def _resolve_json_path(raw: Optional[str], project: Path, fallback: Optional[Path] = None) -> Optional[Path]:
    if raw:
        p = Path(raw)
        if p.exists():
            return p
        if not p.is_absolute():
            q = project / p
            if q.exists():
                return q
    if fallback is not None and fallback.exists():
        return fallback
    return fallback


# -----------------------------------------------------------------------------
# Artifact discovery
# -----------------------------------------------------------------------------

def load_nontransformer_bundle(results_dir: Path, seed: int, key: str) -> Bundle:
    root = results_dir / "non_transformers" / f"seed_{seed}" / key
    metrics_path = root / "metrics.json"
    validation_path = root / "validation_predictions.csv"
    test_path = root / "test_predictions.csv"
    metrics = _read_json(metrics_path)
    validation = _load_prediction(validation_path)
    test = _load_prediction(test_path)
    threshold = _threshold_from_payload(metrics, test, test_path)
    return Bundle(
        key=key,
        display_name=metrics.get("paper_name", _display(key)),
        source_type=metrics.get("paradigm", "Non-Transformer"),
        metrics=metrics,
        validation=validation,
        test=test,
        threshold=threshold,
        metrics_path=metrics_path,
        validation_path=validation_path,
        test_path=test_path,
    )


def _find_transformer_result_dir(project: Path, root: Path, selected: dict) -> Path:
    # Preferred: exact result_dir recorded by the training runner.
    raw = selected.get("result_dir")
    p = _resolve_json_path(raw, project)
    if p is not None and p.is_dir() and (p / "test_predictions.csv").is_file():
        return p

    # Common current-protocol path.
    preferred = root / "full_primary"
    if (preferred / "metrics.json").is_file() and (preferred / "test_predictions.csv").is_file():
        return preferred

    # Last-resort discovery: require a complete prediction bundle.
    candidates = []
    for mp in root.glob("*/metrics.json"):
        d = mp.parent
        if (d / "validation_predictions.csv").is_file() and (d / "test_predictions.csv").is_file():
            candidates.append(d)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No complete result directory found under {root}")
    raise RuntimeError(f"Ambiguous result directories under {root}: {candidates}")


def load_transformer_bundle(project: Path, results_dir: Path, seed: int, key: str) -> Bundle:
    root = results_dir / "models" / "frd" / f"seed_{seed}" / key
    selected_path = root / "selected_result.json"
    if not selected_path.is_file():
        nested = list(root.glob("*/selected_result.json"))
        if len(nested) == 1:
            selected_path = nested[0]
        else:
            raise FileNotFoundError(f"selected_result.json missing for {key}: {root}")
    selected = _read_json(selected_path)
    result_dir = _find_transformer_result_dir(project, root, selected)
    metrics_path = result_dir / "metrics.json"
    metrics = _read_json(metrics_path) if metrics_path.is_file() else selected
    # selected_result is authoritative for selection metadata if metrics omitted it.
    for k in ("selected_threshold", "validation_default", "test_default", "test_calibrated", "model"):
        if k not in metrics and k in selected:
            metrics[k] = selected[k]
    validation_path = result_dir / "validation_predictions.csv"
    test_path = result_dir / "test_predictions.csv"
    validation = _load_prediction(validation_path)
    test = _load_prediction(test_path)
    threshold = _threshold_from_payload(metrics, test, test_path)
    return Bundle(
        key=key,
        display_name=_display(key),
        source_type="Transformer",
        metrics=metrics,
        validation=validation,
        test=test,
        threshold=threshold,
        metrics_path=metrics_path,
        validation_path=validation_path,
        test_path=test_path,
    )


def find_ensemble_manifest(results_dir: Path, seed: int) -> Path:
    preferred = results_dir / "sentiment_extension" / f"seed_{seed}" / "ensembles" / "frd" / "selection_manifest.json"
    if preferred.is_file():
        return preferred
    candidates = list(results_dir.glob(f"**/sentiment_extension/seed_{seed}/ensembles/frd/selection_manifest.json"))
    if not candidates:
        candidates = list(results_dir.glob(f"**/seed_{seed}/ensembles/frd/selection_manifest.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("FRD sentiment-extension ensemble selection_manifest.json not found")
    raise RuntimeError(f"Ambiguous ensemble manifests: {candidates}")


def _resolve_ensemble_method_dir(project: Path, manifest_path: Path, selected: dict, method: str) -> Path:
    key = f"{method}_dir"
    raw = selected.get(key)
    if raw:
        p = _resolve_json_path(raw, project)
        if p and p.is_dir() and (p / "metrics.json").is_file():
            return p
    rank = int(selected["rank"])
    members = list(selected["members"])
    group = f"triple_rank_{rank:02d}__" + "__".join(members)
    fallback = manifest_path.parent / group / method
    if (fallback / "metrics.json").is_file():
        return fallback
    raise FileNotFoundError(f"Could not resolve ensemble directory for rank={rank}, method={method}")


def load_final_ensemble_bundle(project: Path, results_dir: Path, seed: int) -> Tuple[Bundle, Path, dict]:
    manifest_path = find_ensemble_manifest(results_dir, seed)
    manifest = _read_json(manifest_path)
    selected_list = manifest.get("selected", [])
    if not selected_list:
        raise RuntimeError(f"No selected triples in {manifest_path}")
    rank1 = sorted(selected_list, key=lambda x: int(x["rank"]))[0]
    if int(rank1["rank"]) != 1:
        raise RuntimeError("Rank-1 ensemble not present in selection manifest")
    method_dir = _resolve_ensemble_method_dir(project, manifest_path, rank1, "weighted_soft")
    metrics_path = method_dir / "metrics.json"
    validation_path = method_dir / "validation_predictions.csv"
    test_path = method_dir / "test_predictions.csv"
    metrics = _read_json(metrics_path)
    validation = _load_prediction(validation_path)
    test = _load_prediction(test_path)
    threshold = _threshold_from_payload(metrics, test, test_path)
    member_names = [_display(k) for k in rank1.get("members", [])]
    display_name = " + ".join(member_names) + " (Weighted Soft Voting)"
    key = "ensemble_rank1_weighted_soft"
    bundle = Bundle(
        key=key,
        display_name=display_name,
        source_type="Ensemble",
        metrics=metrics,
        validation=validation,
        test=test,
        threshold=threshold,
        metrics_path=metrics_path,
        validation_path=validation_path,
        test_path=test_path,
    )
    return bundle, manifest_path, rank1


# -----------------------------------------------------------------------------
# Metrics and alignment
# -----------------------------------------------------------------------------

def align_frames(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aa = a.rename(columns={"y_true": "y_true_a", "y_prob_cg": "prob_a"})
    bb = b.rename(columns={"y_true": "y_true_b", "y_prob_cg": "prob_b"})
    merged = aa.merge(bb, on="sample_id", how="outer", validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        bad = merged.loc[merged["_merge"] != "both", ["sample_id", "_merge"]].head().to_dict("records")
        raise RuntimeError(f"Sample alignment mismatch: {name_a} vs {name_b}: {bad}")
    if not np.array_equal(merged["y_true_a"].to_numpy(int), merged["y_true_b"].to_numpy(int)):
        raise RuntimeError(f"y_true mismatch: {name_a} vs {name_b}")
    merged = merged.sort_values("sample_id").reset_index(drop=True)
    y = merged["y_true_a"].to_numpy(int)
    pa = merged["prob_a"].to_numpy(float)
    pb = merged["prob_b"].to_numpy(float)
    sid = merged["sample_id"].to_numpy(str)
    return y, pa, pb, sid


def compute_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= float(threshold)).astype(int)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    specificity = tn / max(1, tn + fp)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "errors": int(fp + fn),
    }


def _metric_value(metric: str, y: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    pred = (prob >= float(threshold)).astype(int)
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    if metric == "f1":
        return float(f1_score(y, pred, zero_division=0))
    if metric == "roc_auc":
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, prob))
    raise KeyError(metric)


# -----------------------------------------------------------------------------
# Statistical procedures
# -----------------------------------------------------------------------------

def mcnemar_paired(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    correct_a = pred_a == y
    correct_b = pred_b == y
    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        return {
            "a_correct_b_wrong": 0,
            "a_wrong_b_correct": 0,
            "discordant": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "no_discordant_pairs",
        }
    if discordant < 25:
        p = float(binomtest(min(a_correct_b_wrong, a_wrong_b_correct), discordant, p=0.5, alternative="two-sided").pvalue)
        return {
            "a_correct_b_wrong": a_correct_b_wrong,
            "a_wrong_b_correct": a_wrong_b_correct,
            "discordant": discordant,
            "statistic": float("nan"),
            "p_value": p,
            "method": "exact_two_sided_binomial",
        }
    stat = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1.0) ** 2 / discordant
    p = float(chi2.sf(stat, 1))
    return {
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant": discordant,
        "statistic": float(stat),
        "p_value": p,
        "method": "continuity_corrected_mcnemar",
    }


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted.tolist()


def paired_bootstrap_difference(
    y: np.ndarray,
    prob_a: np.ndarray,
    threshold_a: float,
    prob_b: np.ndarray,
    threshold_b: float,
    metric: str,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap difference B-A using identical resampled indices."""
    y = np.asarray(y, dtype=int)
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)
    point_a = _metric_value(metric, y, prob_a, threshold_a)
    point_b = _metric_value(metric, y, prob_b, threshold_b)
    point_diff = point_b - point_a
    if n_boot <= 0:
        return {
            "metric": metric,
            "a": point_a,
            "b": point_b,
            "difference_b_minus_a": point_diff,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_n_valid": 0,
            "ci_excludes_zero": False,
        }
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        # AUC can be undefined if a bootstrap sample contains one class. With
        # n~8k balanced this is effectively impossible, but guard anyway.
        if metric == "roc_auc" and len(np.unique(yy)) < 2:
            continue
        va = _metric_value(metric, yy, prob_a[idx], threshold_a)
        vb = _metric_value(metric, yy, prob_b[idx], threshold_b)
        if math.isfinite(va) and math.isfinite(vb):
            diffs.append(vb - va)
    if not diffs:
        lo = hi = float("nan")
    else:
        arr = np.asarray(diffs, dtype=float)
        lo = float(np.quantile(arr, alpha / 2))
        hi = float(np.quantile(arr, 1 - alpha / 2))
    excludes = bool(math.isfinite(lo) and math.isfinite(hi) and (lo > 0 or hi < 0))
    return {
        "metric": metric,
        "a": point_a,
        "b": point_b,
        "difference_b_minus_a": point_diff,
        "difference_pp_b_minus_a": point_diff * 100.0 if metric in {"accuracy", "f1"} else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "ci_low_pp": lo * 100.0 if metric in {"accuracy", "f1"} and math.isfinite(lo) else float("nan"),
        "ci_high_pp": hi * 100.0 if metric in {"accuracy", "f1"} and math.isfinite(hi) else float("nan"),
        "bootstrap_n_valid": len(diffs),
        "ci_excludes_zero": excludes,
    }


def bootstrap_ci_single(
    y: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    metric: str,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict:
    point = _metric_value(metric, y, prob, threshold)
    if n_boot <= 0:
        return {"metric": metric, "estimate": point, "ci_low": float("nan"), "ci_high": float("nan"), "bootstrap_n_valid": 0}
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if metric == "roc_auc" and len(np.unique(yy)) < 2:
            continue
        value = _metric_value(metric, yy, prob[idx], threshold)
        if math.isfinite(value):
            vals.append(value)
    arr = np.asarray(vals, dtype=float)
    return {
        "metric": metric,
        "estimate": point,
        "ci_low": float(np.quantile(arr, alpha / 2)) if len(arr) else float("nan"),
        "ci_high": float(np.quantile(arr, 1 - alpha / 2)) if len(arr) else float("nan"),
        "bootstrap_n_valid": len(arr),
    }


# -----------------------------------------------------------------------------
# Selection and reporting
# -----------------------------------------------------------------------------

def select_best_validation(bundles: Iterable[Bundle]) -> Bundle:
    items = list(bundles)
    if not items:
        raise RuntimeError("No candidates available for validation-only selection")
    return max(items, key=_validation_tuple)


def select_backbone_representatives(transformers: Dict[str, Bundle]) -> Tuple[Dict[str, Bundle], pd.DataFrame]:
    selected: Dict[str, Bundle] = {}
    rows = []
    for family, (plain_key, sent_key) in BACKBONE_PAIRS.items():
        if plain_key not in transformers or sent_key not in transformers:
            raise RuntimeError(f"Missing matched candidates for {family}: {plain_key}, {sent_key}")
        choices = [transformers[plain_key], transformers[sent_key]]
        winner = select_best_validation(choices)
        selected[family] = winner
        for b in choices:
            vm = b.metrics.get("validation_default", {})
            rows.append(
                {
                    "backbone": family,
                    "candidate_key": b.key,
                    "candidate": b.display_name,
                    "selected": b.key == winner.key,
                    "validation_accuracy": vm.get("accuracy"),
                    "validation_f1": vm.get("f1"),
                    "validation_roc_auc": vm.get("roc_auc"),
                    "selected_threshold": b.threshold,
                }
            )
    return selected, pd.DataFrame(rows)


def _stage_row(order: int, stage: str, bundle: Bundle) -> dict:
    y = bundle.test["y_true"].to_numpy(int)
    p = bundle.test["y_prob_cg"].to_numpy(float)
    tm = compute_metrics(y, p, bundle.threshold)
    vm = bundle.metrics.get("validation_default", {})
    return {
        "stage_order": order,
        "stage": stage,
        "model_key": bundle.key,
        "model": bundle.display_name,
        "source_type": bundle.source_type,
        "selection_basis": "validation_only",
        "validation_accuracy": vm.get("accuracy"),
        "validation_f1": vm.get("f1"),
        "validation_roc_auc": vm.get("roc_auc"),
        "selected_threshold": bundle.threshold,
        "test_n": len(y),
        **tm,
    }


def _paired_comparison_row(label: str, stage_a: str, a: Bundle, stage_b: str, b: Bundle) -> dict:
    y, pa, pb, _ = align_frames(a.test, b.test, a.display_name, b.display_name)
    pred_a = (pa >= a.threshold).astype(int)
    pred_b = (pb >= b.threshold).astype(int)
    mc = mcnemar_paired(y, pred_a, pred_b)
    ma = compute_metrics(y, pa, a.threshold)
    mb = compute_metrics(y, pb, b.threshold)
    return {
        "comparison": label,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "model_a_key": a.key,
        "model_a": a.display_name,
        "model_b_key": b.key,
        "model_b": b.display_name,
        "threshold_a": a.threshold,
        "threshold_b": b.threshold,
        "n": len(y),
        "accuracy_a": ma["accuracy"],
        "accuracy_b": mb["accuracy"],
        "accuracy_delta_pp_b_minus_a": 100.0 * (mb["accuracy"] - ma["accuracy"]),
        "f1_a": ma["f1"],
        "f1_b": mb["f1"],
        "f1_delta_pp_b_minus_a": 100.0 * (mb["f1"] - ma["f1"]),
        "roc_auc_a": ma["roc_auc"],
        "roc_auc_b": mb["roc_auc"],
        "roc_auc_delta_b_minus_a": mb["roc_auc"] - ma["roc_auc"],
        **mc,
    }


def _bootstrap_rows(label: str, stage_a: str, a: Bundle, stage_b: str, b: Bundle, n_boot: int, seed: int) -> List[dict]:
    y, pa, pb, _ = align_frames(a.test, b.test, a.display_name, b.display_name)
    rows = []
    for metric in ("accuracy", "f1", "roc_auc"):
        d = paired_bootstrap_difference(y, pa, a.threshold, pb, b.threshold, metric, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "comparison": label,
                "stage_a": stage_a,
                "stage_b": stage_b,
                "model_a_key": a.key,
                "model_a": a.display_name,
                "model_b_key": b.key,
                "model_b": b.display_name,
                **d,
            }
        )
    return rows


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def _fmt_p(x: float) -> str:
    x = float(x)
    if x < 1e-4:
        return f"{x:.3e}"
    return f"{x:.5f}"


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def run(args) -> int:
    project = args.project.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve() if args.results_dir else project / "results"
    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else results_dir / "paper_pack" / "final_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("FRD FINAL CROSS-PARADIGM PAIRED STATISTICAL ANALYSIS")
    print(f"Runner: {RUNNER_VERSION}")
    print(f"Project: {project}")
    print(f"Results: {results_dir}")
    print(f"Output:  {out_dir}")
    print(f"Seed: {args.seed} | Bootstrap: {args.bootstrap} | Bootstrap seed: {args.bootstrap_seed}")
    print("=" * 88)

    # Load all non-Transformer bundles.
    nontr: Dict[str, Bundle] = {}
    missing = []
    for key in TRADITIONAL_KEYS + DEEP_KEYS:
        try:
            nontr[key] = load_nontransformer_bundle(results_dir, args.seed, key)
        except Exception as exc:
            missing.append(f"{key}: {exc!r}")
    if missing:
        raise RuntimeError("Missing/incomplete non-Transformer artifacts:\n  " + "\n  ".join(missing))

    # Load all 14 Transformer bundles.
    transformers: Dict[str, Bundle] = {}
    missing = []
    for key in PLAIN_TRANSFORMERS + SENTIMENT_TRANSFORMERS:
        try:
            transformers[key] = load_transformer_bundle(project, results_dir, args.seed, key)
        except Exception as exc:
            missing.append(f"{key}: {exc!r}")
    if missing:
        raise RuntimeError("Missing/incomplete Transformer artifacts:\n  " + "\n  ".join(missing))

    ensemble, ensemble_manifest_path, ensemble_rank1 = load_final_ensemble_bundle(project, results_dir, args.seed)

    # Validation-only selections.
    traditional = select_best_validation(nontr[k] for k in TRADITIONAL_KEYS)
    deep = select_best_validation(nontr[k] for k in DEEP_KEYS)
    plain_transformer = select_best_validation(transformers[k] for k in PLAIN_TRANSFORMERS)
    reps_by_family, rep_df = select_backbone_representatives(transformers)
    extended_individual = select_best_validation(reps_by_family.values())

    rep_df.to_csv(out_dir / "transformer_representative_selection.csv", index=False)

    stages: List[Tuple[str, Bundle]] = [
        ("Traditional ML", traditional),
        ("Deep Learning", deep),
        ("Plain Transformer", plain_transformer),
        ("Validation-selected Transformer configuration", extended_individual),
        ("Final weighted ensemble", ensemble),
    ]

    # Validate that all stage bundles use the exact same held-out test observations.
    base = stages[0][1].test
    for stage_name, b in stages[1:]:
        align_frames(base, b.test, stages[0][1].display_name, b.display_name)

    # Stage metrics.
    stage_rows = [_stage_row(i + 1, name, bundle) for i, (name, bundle) in enumerate(stages)]
    stage_df = pd.DataFrame(stage_rows)
    stage_df["accuracy_delta_pp_from_previous"] = stage_df["accuracy"].diff() * 100.0
    stage_df["f1_delta_pp_from_previous"] = stage_df["f1"].diff() * 100.0
    stage_df["roc_auc_delta_from_previous"] = stage_df["roc_auc"].diff()
    stage_df.to_csv(out_dir / "stage_selection_and_metrics.csv", index=False)

    # Stage-level bootstrap confidence intervals.
    ci_rows = []
    for stage_name, bundle in stages:
        y = bundle.test["y_true"].to_numpy(int)
        p = bundle.test["y_prob_cg"].to_numpy(float)
        for metric in ("accuracy", "f1", "roc_auc"):
            ci = bootstrap_ci_single(y, p, bundle.threshold, metric, args.bootstrap, args.bootstrap_seed)
            ci_rows.append({
                "stage": stage_name,
                "model_key": bundle.key,
                "model": bundle.display_name,
                **ci,
            })
    pd.DataFrame(ci_rows).to_csv(out_dir / "stage_confidence_intervals.csv", index=False)

    # Planned adjacent progression comparisons. The family is predeclared and
    # Holm-adjusted together.
    progression_pairs = [
        ("Traditional ML -> Deep Learning", stages[0], stages[1]),
        ("Deep Learning -> Plain Transformer", stages[1], stages[2]),
        ("Plain Transformer -> Validation-selected Transformer configuration", stages[2], stages[3]),
        ("Validation-selected Transformer configuration -> Final weighted ensemble", stages[3], stages[4]),
    ]
    prog_test_rows = []
    prog_boot_rows = []
    for i, (label, (stage_a, a), (stage_b, b)) in enumerate(progression_pairs):
        prog_test_rows.append(_paired_comparison_row(label, stage_a, a, stage_b, b))
        prog_boot_rows.extend(_bootstrap_rows(label, stage_a, a, stage_b, b, args.bootstrap, args.bootstrap_seed))
    adj = holm_adjust([r["p_value"] for r in prog_test_rows])
    for row, hp in zip(prog_test_rows, adj):
        row["holm_p_progression_family"] = hp
        row["significant_after_holm_0_05"] = bool(hp < 0.05)
    pd.DataFrame(prog_test_rows).to_csv(out_dir / "planned_progression_paired_tests.csv", index=False)
    pd.DataFrame(prog_boot_rows).to_csv(out_dir / "planned_progression_bootstrap_effects.csv", index=False)

    # Matched plain-vs-sentiment ablations. This is a separate inferential family.
    sent_test_rows = []
    sent_boot_rows = []
    for family, (plain_key, sent_key) in BACKBONE_PAIRS.items():
        a = transformers[plain_key]
        b = transformers[sent_key]
        label = f"{family}: plain vs sentiment"
        row = _paired_comparison_row(label, family + " plain", a, family + " sentiment", b)
        row["backbone"] = family
        sent_test_rows.append(row)
        for br in _bootstrap_rows(label, family + " plain", a, family + " sentiment", b, args.bootstrap, args.bootstrap_seed):
            br["backbone"] = family
            sent_boot_rows.append(br)
    sent_adj = holm_adjust([r["p_value"] for r in sent_test_rows])
    for row, hp in zip(sent_test_rows, sent_adj):
        row["holm_p_sentiment_family"] = hp
        row["significant_after_holm_0_05"] = bool(hp < 0.05)
    pd.DataFrame(sent_test_rows).to_csv(out_dir / "sentiment_ablation_paired_tests.csv", index=False)
    pd.DataFrame(sent_boot_rows).to_csv(out_dir / "sentiment_ablation_bootstrap_effects.csv", index=False)

    # Convenient standalone file for the final system contrast. This is already
    # present as the last planned progression comparison, but is surfaced
    # separately because the paper discusses the ensemble-vs-individual gain.
    final_comp = prog_test_rows[-1].copy()
    final_boot = [r for r in prog_boot_rows if r["comparison"] == progression_pairs[-1][0]]
    final_payload = {
        "comparison": final_comp,
        "paired_bootstrap": final_boot,
        "interpretation_note": (
            "The comparator is the strongest individual Transformer configuration selected exclusively "
            "from validation metrics. This is preferable for inference to selecting a comparator by held-out test performance."
        ),
    }
    _json_dump(out_dir / "ensemble_vs_selected_individual.json", final_payload)
    pd.DataFrame([final_comp]).to_csv(out_dir / "ensemble_vs_selected_individual.csv", index=False)

    # Selection audit for reviewer/reproducibility traceability.
    audit = {
        "runner_version": RUNNER_VERSION,
        "seed": args.seed,
        "bootstrap_n": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "selection_rule": "validation_default accuracy, then F1, then ROC-AUC, then deterministic model key tie-break",
        "test_selection_forbidden": True,
        "positive_class": "CG / machine-generated = 1",
        "traditional_selected": {"key": traditional.key, "model": traditional.display_name},
        "deep_learning_selected": {"key": deep.key, "model": deep.display_name},
        "plain_transformer_selected": {"key": plain_transformer.key, "model": plain_transformer.display_name},
        "transformer_representatives_by_backbone": {k: v.key for k, v in reps_by_family.items()},
        "extended_individual_selected": {"key": extended_individual.key, "model": extended_individual.display_name},
        "ensemble_manifest": str(ensemble_manifest_path),
        "ensemble_rank": int(ensemble_rank1.get("rank", 1)),
        "ensemble_members": list(ensemble_rank1.get("members", [])),
        "ensemble_method": "weighted_soft",
        "ensemble_result_dir": str(ensemble.metrics_path.parent),
        "multiple_testing_families": {
            "planned_progression": len(prog_test_rows),
            "matched_sentiment_ablation": len(sent_test_rows),
        },
        "notes": [
            "All model/stage representatives are selected using validation information only.",
            "All paired tests use identical held-out FRD test observations.",
            "Each system retains its own validation-selected operating threshold.",
            "The plain-Transformer -> extended-individual progression contrast is system-level and does not isolate sentiment causally.",
            "Use the seven matched plain-vs-sentiment comparisons for sentiment-specific inference.",
        ],
    }
    _json_dump(out_dir / "selection_audit.json", audit)

    # Human-readable summary.
    lines = [
        "# Final FRD Statistical Analysis",
        "",
        "All representatives were selected from validation results only; the held-out FRD test set was used only for final metrics and paired inference.",
        "",
        "## Validation-selected performance progression",
        "",
        "| Stage | Selected system | Accuracy | F1 | ROC-AUC |",
        "|---|---|---:|---:|---:|",
    ]
    for row in stage_rows:
        lines.append(
            f"| {row['stage']} | {row['model']} | {_fmt_pct(row['accuracy'])} | {_fmt_pct(row['f1'])} | {row['roc_auc']:.4f} |"
        )
    lines += ["", "## Planned adjacent paired comparisons", "", "| Comparison | ΔAccuracy (pp) | McNemar p | Holm p | Significant |", "|---|---:|---:|---:|---|"]
    for r in prog_test_rows:
        lines.append(
            f"| {r['comparison']} | {r['accuracy_delta_pp_b_minus_a']:+.3f} | {_fmt_p(r['p_value'])} | {_fmt_p(r['holm_p_progression_family'])} | {'yes' if r['significant_after_holm_0_05'] else 'no'} |"
        )
    lines += [
        "",
        "## Interpretation note",
        "",
        "The progression analysis quantifies system-level gains. The transition from the plain Transformer stage to the validation-selected Transformer configuration may combine backbone selection and optional sentiment fusion; it must not be interpreted as a causal sentiment effect. Sentiment-specific claims should use the matched seven-backbone ablation family.",
        "",
        "## Matched sentiment ablations",
        "",
        "| Backbone | ΔAccuracy (pp, sentiment - plain) | McNemar p | Holm p | Significant |",
        "|---|---:|---:|---:|---|",
    ]
    for r in sent_test_rows:
        lines.append(
            f"| {r['backbone']} | {r['accuracy_delta_pp_b_minus_a']:+.3f} | {_fmt_p(r['p_value'])} | {_fmt_p(r['holm_p_sentiment_family'])} | {'yes' if r['significant_after_holm_0_05'] else 'no'} |"
        )
    lines += [
        "",
        "Paired bootstrap files report B-A effect sizes and 95% percentile confidence intervals for Accuracy, F1, and ROC-AUC.",
    ]
    (out_dir / "final_statistics_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # Console summary.
    print("\n[VALIDATION-SELECTED STAGES]")
    for row in stage_rows:
        print(f"  {row['stage']:<46} {row['model']:<75} Acc={row['accuracy']:.6f} F1={row['f1']:.6f} AUC={row['roc_auc']:.6f}")
    print("\n[PLANNED PAIRED TESTS]")
    for r in prog_test_rows:
        print(
            f"  {r['comparison']}: ΔAcc={r['accuracy_delta_pp_b_minus_a']:+.3f} pp, "
            f"McNemar p={r['p_value']:.6g}, Holm p={r['holm_p_progression_family']:.6g}, "
            f"significant={r['significant_after_holm_0_05']}"
        )
    print("\n[MATCHED SENTIMENT ABLATIONS]")
    for r in sent_test_rows:
        print(
            f"  {r['backbone']}: ΔAcc={r['accuracy_delta_pp_b_minus_a']:+.3f} pp, "
            f"Holm p={r['holm_p_sentiment_family']:.6g}, significant={r['significant_after_holm_0_05']}"
        )
    print(f"\nComplete. Paper-ready outputs: {out_dir}")
    return 0


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--project", type=Path, default=here, help="FRD project root")
    ap.add_argument("--results-dir", type=Path, default=None, help="Override results directory")
    ap.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=1000, help="Paired bootstrap resamples; 0 disables CIs")
    ap.add_argument("--bootstrap-seed", type=int, default=314159)
    return ap.parse_args(argv)


if __name__ == "__main__":
    try:
        sys.exit(run(parse_args()))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n[FATAL] {exc!r}", file=sys.stderr)
        raise
