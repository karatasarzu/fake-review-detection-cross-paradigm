# -*- coding: utf-8 -*-
"""
FRD -> AiGen-FoodReview External Validation
============================================

Purpose
-------
Zero-shot external validation of the already-trained FRD Transformer systems on
AiGen-FoodReview's official test.csv.  This script NEVER trains/fine-tunes a
model, NEVER fits a new sentiment scaler, NEVER selects a threshold on AiGen,
and NEVER changes ensemble membership or weights.

The FRD-side development decisions are reused exactly:
  * model parameters/checkpoints: trained on FRD train
  * decision thresholds: selected on FRD validation
  * plain-vs-sentiment representatives: selected on FRD validation
  * ensemble membership/weights: selected on FRD validation
  * sentiment scaler parameters: stored in each FRD hybrid artifact

Expected project layout
-----------------------
Place this file in the same project directory that contains the completed
``results`` folder from the FRD study, and place AiGen-FoodReview ``test.csv``
next to it (or pass --dataset).

Typical commands (Windows)
--------------------------
  .\\.venv_frd_safe\\Scripts\\python.exe FRD_AiGen_External_Validation.py --scope all
  .\\.venv_frd_safe\\Scripts\\python.exe FRD_AiGen_External_Validation.py diagnostics --frd-dataset "fake reviews dataset.csv"
  .\\.venv_frd_safe\\Scripts\\python.exe FRD_AiGen_External_Validation.py device-check

Sub-commands (post-hoc, read stored predictions; no training or selection)
  diagnostics   truncation rates, surface features, format-restricted and
                length-stratified ROC-AUC, operating-point checks
  device-check  re-scores the FRD test set with ModernBERT on CPU and compares
                with the stored GPU predictions

  .\\.venv_frd_safe\\Scripts\\python.exe FRD_AiGen_External_Validation.py --dry-run
  .\\.venv_frd_safe\\Scripts\\python.exe FRD_AiGen_External_Validation.py

Alternative venv:
  .\\.venv\\Scripts\\python.exe FRD_AiGen_External_Validation.py

Outputs
-------
  results/external_aigen/seed_42/
      dataset_audit.json
      aigen_sentiment_features.csv
      models/<model_key>/predictions.csv
      models/<model_key>/metrics.json
      ensembles/<triple>/<equal_or_weighted>/predictions.csv
      ensembles/<triple>/<equal_or_weighted>/metrics.json
      all_systems_primary_metrics.csv
      all_systems_default_05_metrics.csv
      confidence_intervals.csv
      paper_table_selected_representatives_plus_final.csv
      final_system.json
      run_summary.json

AiGen-FoodReview official label mapping
---------------------------------------
  label=0 -> authentic / human-written
  label=1 -> machine-generated

This matches the FRD convention used by the current study:
  label_num=0 -> original (OR)
  label_num=1 -> computer-generated (CG)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

RUNNER_VERSION = "1.0.3"
OFFICIAL_AIGEN_TEST_MD5 = "c41a361eafa9800e3e8be3362a0101f4"
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAP_SEED = 314159

FEATURE_NAMES = (
    "polarity",
    "subjectivity",
    "sentence_polarity_mean",
    "sentence_polarity_std",
    "sentence_polarity_min",
    "sentence_polarity_max",
    "positive_sentence_ratio",
    "negative_sentence_ratio",
    "neutral_sentence_ratio",
)

# All non-large Transformer systems in the final FRD sentiment-extension study.
MODEL_DISPLAY_NAMES = OrderedDict([
    ("bert", "BERT"),
    ("bert_sentiment_profile", "BERT + Sentiment Profile"),
    ("roberta", "RoBERTa"),
    ("roberta_sentiment_profile", "RoBERTa + Sentiment Profile"),
    ("deberta", "DeBERTa"),
    ("deberta_sentiment_profile", "DeBERTa + Sentiment Profile"),
    ("deberta_v3_base", "DeBERTa-v3"),
    ("deberta_v3_sentiment_gated", "DeBERTa-v3 + Sentiment Profile (Gated)"),
    ("modernbert_base", "ModernBERT"),
    ("modernbert_sentiment_profile", "ModernBERT + Sentiment Profile"),
    ("electra_base", "ELECTRA"),
    ("electra_sentiment_profile", "ELECTRA + Sentiment Profile"),
    ("distilbert", "DistilBERT"),
    ("distilbert_sentiment_profile", "DistilBERT + Sentiment Profile"),
])

BACKBONE_CANDIDATES = OrderedDict([
    ("bert", ("bert", "bert_sentiment_profile")),
    ("roberta", ("roberta", "roberta_sentiment_profile")),
    ("deberta", ("deberta", "deberta_sentiment_profile")),
    ("deberta_v3", ("deberta_v3_base", "deberta_v3_sentiment_gated")),
    ("modernbert", ("modernbert_base", "modernbert_sentiment_profile")),
    ("electra", ("electra_base", "electra_sentiment_profile")),
    ("distilbert", ("distilbert", "distilbert_sentiment_profile")),
])


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _csv_atomic(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        return float(x)
    except Exception:
        return default


def _patch_torch_compile_for_windows() -> None:
    """Disable torch.compile decorators during Windows inference.

    ModernBERT in Transformers 4.48.x decorates a few methods with
    ``torch.compile`` at import time. PyTorch 2.5 raises on Windows before
    the model can even be loaded. The FRD checkpoints were already trained;
    for pure inference compilation is not required. Replacing torch.compile
    with an identity decorator preserves eager-mode semantics and avoids the
    Windows-only import failure.
    """
    if os.name != "nt":
        return
    try:
        import torch
    except Exception:
        return
    if not hasattr(torch, "compile"):
        return
    if getattr(torch.compile, "_frd_windows_identity_patch", False):
        return

    def _identity_compile(model=None, *args, **kwargs):
        if model is None:
            def _decorator(fn):
                return fn
            return _decorator
        return model

    _identity_compile._frd_windows_identity_patch = True
    torch.compile = _identity_compile


def _device_from_arg(value: str):
    import torch
    value = str(value).lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return value


def _cleanup_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def _is_modernbert_key(key: str) -> bool:
    return str(key).startswith("modernbert")


def _is_deberta_v3_key(key: str) -> bool:
    return str(key).startswith("deberta_v3")


def _load_artifact_tokenizer(path: Path, key: str):
    """Load the stored tokenizer without unnecessary fast-tokenizer conversion.

    DeBERTa-v3 uses a SentencePiece tokenizer.  Some older Windows
    environments have an incompatible sentencepiece/protobuf pair that fails
    only while Transformers converts the slow tokenizer to the fast backend
    (``google.protobuf.internal.builder`` ImportError).  The slow tokenizer is
    fully sufficient for inference and consumes the exact same stored
    SentencePiece model, so prefer it for DeBERTa-v3.
    """
    from transformers import AutoTokenizer
    if _is_deberta_v3_key(key):
        try:
            return AutoTokenizer.from_pretrained(path, use_fast=False)
        except Exception as slow_exc:
            print(f"[WARN] {key}: slow tokenizer load failed: {slow_exc}")
            print(f"[WARN] {key}: retrying tokenizer with the fast backend...")
            return AutoTokenizer.from_pretrained(path, use_fast=True)
    return AutoTokenizer.from_pretrained(path)


def _modernbert_config_from_artifact(path: Path):
    """Load ModernBERT config in a conservative eager/FP32 inference mode."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(path)
    if hasattr(cfg, "reference_compile"):
        cfg.reference_compile = False
    if hasattr(cfg, "deterministic_flash_attn"):
        cfg.deterministic_flash_attn = True
    return cfg


def batch_size_fallbacks(initial: int) -> list[int]:
    cur = max(1, int(initial))
    out: list[int] = []
    while True:
        if cur not in out:
            out.append(cur)
        if cur <= 1:
            break
        cur = max(1, cur // 2)
    return out


# -----------------------------------------------------------------------------
# Dataset loading and integrity audit
# -----------------------------------------------------------------------------


def load_aigen_test(path: Path):
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"AiGen test CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"ID", "text", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"AiGen test.csv missing required columns: {missing}")
    if df["ID"].isna().any() or df["ID"].duplicated().any():
        raise RuntimeError("AiGen ID column contains missing or duplicate IDs")
    if df["text"].isna().any():
        raise RuntimeError("AiGen text column contains missing values")

    # Official AiGen-FoodReview mapping: 0=authentic, 1=machine-generated.
    labels = pd.to_numeric(df["label"], errors="raise").astype(int)
    if set(labels.unique()) != {0, 1}:
        raise RuntimeError(f"Expected AiGen labels {{0,1}}, got {sorted(labels.unique().tolist())}")

    out = pd.DataFrame({
        "source_index": df["ID"].to_numpy(),
        "text": df["text"].astype(str).to_numpy(),
        "label_num": labels.to_numpy(dtype=int),
    })
    return df, out


def dataset_audit(dataset_path: Path, raw_df, frame, results_dir: Path) -> dict:
    import pandas as pd

    norms = frame["text"].map(_norm_text)
    raw_dups = int(frame["text"].duplicated(keep=False).sum())
    norm_dups = int(norms.duplicated(keep=False).sum())
    class_counts = frame["label_num"].value_counts().sort_index().to_dict()

    audit: dict[str, Any] = {
        "dataset": "AiGen-FoodReview official test split",
        "dataset_path": str(Path(dataset_path).resolve()),
        "rows": int(len(frame)),
        "columns_in_source_file": int(raw_df.shape[1]),
        "class_counts": {
            "0_authentic": int(class_counts.get(0, 0)),
            "1_machine_generated": int(class_counts.get(1, 0)),
        },
        "missing_text": int(frame["text"].isna().sum()),
        "duplicate_ids": int(frame["source_index"].duplicated().sum()),
        "rows_in_raw_exact_duplicate_groups": raw_dups,
        "rows_in_normalized_duplicate_groups": norm_dups,
        "md5": _md5(Path(dataset_path)),
        "sha256": _sha256(Path(dataset_path)),
        "official_test_md5_expected": OFFICIAL_AIGEN_TEST_MD5,
        "official_test_md5_match": _md5(Path(dataset_path)) == OFFICIAL_AIGEN_TEST_MD5,
        "label_mapping": {"0": "authentic/human-written", "1": "machine-generated"},
        "frd_overlap": None,
    }

    # Strong external-validity hygiene: audit normalized exact-text overlap with
    # the persisted FRD benchmark if it is available. No rows are removed here;
    # the audit simply makes any overlap explicit.
    split_path = Path(results_dir) / "splits" / "frd_split.csv"
    if split_path.is_file():
        try:
            frd = pd.read_csv(split_path)
            text_col = "text_" if "text_" in frd.columns else "text" if "text" in frd.columns else None
            if text_col:
                aigen_set = set(norms.tolist())
                overlap_detail = {}
                if "split" in frd.columns:
                    for split_name, g in frd.groupby("split"):
                        count = int(g[text_col].astype(str).map(_norm_text).isin(aigen_set).sum())
                        overlap_detail[str(split_name)] = count
                all_count = int(frd[text_col].astype(str).map(_norm_text).isin(aigen_set).sum())
                audit["frd_overlap"] = {
                    "frd_split_path": str(split_path.resolve()),
                    "normalized_exact_overlap_rows_in_frd": all_count,
                    "by_frd_split": overlap_detail,
                }
        except Exception as exc:
            audit["frd_overlap"] = {"audit_error": repr(exc)}
    return audit


# -----------------------------------------------------------------------------
# Sentiment features: intentionally identical to the current FRD pipeline
# -----------------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|[\r\n]+", str(text)) if p.strip()]
    return parts or [str(text)]


def compute_sentiment_profile(text: str) -> OrderedDict:
    import numpy as np
    from textblob import TextBlob

    blob = TextBlob(str(text)).sentiment
    vals = np.array([TextBlob(s).sentiment.polarity for s in _sentences(text)], dtype=float)
    eps = 0.05
    return OrderedDict([
        ("polarity", float(blob.polarity)),
        ("subjectivity", float(blob.subjectivity)),
        ("sentence_polarity_mean", float(vals.mean())),
        ("sentence_polarity_std", float(vals.std())),
        ("sentence_polarity_min", float(vals.min())),
        ("sentence_polarity_max", float(vals.max())),
        ("positive_sentence_ratio", float((vals > eps).mean())),
        ("negative_sentence_ratio", float((vals < -eps).mean())),
        ("neutral_sentence_ratio", float((np.abs(vals) <= eps).mean())),
    ])


def build_sentiment_features(frame, cache_path: Path, dataset_sha256: str):
    import pandas as pd

    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".meta.json")
    if cache_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cached = pd.read_csv(cache_path)
            if (
                meta.get("dataset_sha256") == dataset_sha256
                and int(meta.get("rows", -1)) == len(frame)
                and list(cached.columns) == list(FEATURE_NAMES)
                and len(cached) == len(frame)
            ):
                print(f"[CACHE] Reusing sentiment features: {cache_path}")
                return cached
        except Exception:
            pass

    print(f"[FEATURES] Computing 9-D sentiment profile for {len(frame):,} reviews...")
    start = time.perf_counter()
    rows = []
    texts = frame["text"].astype(str).tolist()
    report_every = max(100, len(texts) // 20)
    for i, text in enumerate(texts, 1):
        rows.append(compute_sentiment_profile(text))
        if i % report_every == 0 or i == len(texts):
            print(f"  sentiment: {i:,}/{len(texts):,}")
    features = pd.DataFrame(rows, columns=list(FEATURE_NAMES))
    _csv_atomic(features, cache_path)
    _json_dump(meta_path, {
        "dataset_sha256": dataset_sha256,
        "rows": len(frame),
        "feature_names": list(FEATURE_NAMES),
        "seconds": time.perf_counter() - start,
        "note": "Raw AiGen sentiment features only. No scaler was fitted on AiGen.",
    })
    return features


# -----------------------------------------------------------------------------
# FRD artifact discovery
# -----------------------------------------------------------------------------


@dataclass
class ModelBundle:
    key: str
    root: Path
    artifact: Path
    result_dir: Path
    selected_result: dict
    source_metrics: dict
    source_threshold: float
    artifact_meta: dict


def _existing_path_from_json(value: Any, project_root: Path) -> Path | None:
    if not value:
        return None
    p = Path(str(value))
    if p.exists():
        return p
    # If an absolute path stored on the original run no longer resolves after a
    # project move, do not trust it. Relative fallback discovery happens below.
    if not p.is_absolute():
        q = project_root / p
        if q.exists():
            return q
    return None


def resolve_model_bundle(results_dir: Path, seed: int, key: str) -> ModelBundle:
    model_root = Path(results_dir) / "models" / "frd" / f"seed_{seed}" / key
    sel_path = model_root / "selected_result.json"
    if not sel_path.is_file():
        raise FileNotFoundError(f"Missing selected_result.json for {key}: {sel_path}")
    selected = json.loads(sel_path.read_text(encoding="utf-8"))
    project_root = Path(results_dir).parent

    # Resolve selected result directory robustly even if the project was moved.
    result_candidates: list[Path] = []
    p = _existing_path_from_json(selected.get("result_dir"), project_root)
    if p:
        result_candidates.append(p)
    profile = selected.get("profile")
    if profile:
        result_candidates.append(model_root / str(profile))
    for mp in model_root.glob("*/metrics.json"):
        result_candidates.append(mp.parent)

    result_dir = next((p for p in result_candidates if (p / "metrics.json").is_file()), None)
    if result_dir is None:
        raise FileNotFoundError(f"Could not resolve selected FRD result directory for {key} under {model_root}")

    artifact_candidates: list[Path] = []
    p = _existing_path_from_json(selected.get("artifact_path"), project_root)
    if p:
        artifact_candidates.append(p)
    artifact_candidates.append(result_dir / "artifact")
    if profile:
        artifact_candidates.append(model_root / str(profile) / "artifact")
    for ap in model_root.glob("*/artifact/artifact_meta.json"):
        artifact_candidates.append(ap.parent)

    artifact = next((p for p in artifact_candidates if p.is_dir() and (p / "artifact_meta.json").is_file()), None)
    if artifact is None:
        raise FileNotFoundError(f"Could not resolve FRD model artifact for {key} under {model_root}")

    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    threshold = _safe_float(metrics.get("selected_threshold"), _safe_float(selected.get("selected_threshold"), 0.5))
    if threshold is None:
        threshold = 0.5
    meta = json.loads((artifact / "artifact_meta.json").read_text(encoding="utf-8"))
    return ModelBundle(
        key=key,
        root=model_root,
        artifact=artifact,
        result_dir=result_dir,
        selected_result=selected,
        source_metrics=metrics,
        source_threshold=float(threshold),
        artifact_meta=meta,
    )


def discover_all_bundles(results_dir: Path, seed: int) -> tuple[dict[str, ModelBundle], dict[str, str]]:
    found: dict[str, ModelBundle] = {}
    missing: dict[str, str] = {}
    for key in MODEL_DISPLAY_NAMES:
        try:
            found[key] = resolve_model_bundle(results_dir, seed, key)
        except Exception as exc:
            missing[key] = str(exc)
    return found, missing


def find_representative_selection(results_dir: Path, seed: int, bundles: dict[str, ModelBundle]) -> dict:
    preferred = Path(results_dir) / "sentiment_extension" / f"seed_{seed}" / "representatives" / "frd" / "representative_selection.json"
    if preferred.is_file():
        return json.loads(preferred.read_text(encoding="utf-8"))

    # Fallback: reproduce the original validation-only representative rule from
    # stored FRD validation metrics. No AiGen labels are involved.
    selected: dict[str, str] = {}
    for backbone, choices in BACKBONE_CANDIDATES.items():
        available = [c for c in choices if c in bundles]
        if not available:
            continue
        best = available[0]
        bm = bundles[best].source_metrics.get("validation_default", {})
        best_score = tuple(float(bm.get(k, float("-inf"))) for k in ("accuracy", "f1", "roc_auc"))
        for cand in available[1:]:
            cm = bundles[cand].source_metrics.get("validation_default", {})
            score = tuple(float(cm.get(k, float("-inf"))) for k in ("accuracy", "f1", "roc_auc"))
            if score > best_score:
                best, best_score = cand, score
        selected[backbone] = best
    return {
        "protocol": "frd",
        "selection_data": "validation_only",
        "selected_by_backbone": selected,
        "representatives": list(selected.values()),
        "reconstructed_from_stored_frd_validation_metrics": True,
    }


def find_ensemble_manifest(results_dir: Path, seed: int) -> Path | None:
    preferred = Path(results_dir) / "sentiment_extension" / f"seed_{seed}" / "ensembles" / "frd" / "selection_manifest.json"
    if preferred.is_file():
        return preferred
    candidates = list(Path(results_dir).glob(f"**/seed_{seed}/ensembles/frd/selection_manifest.json"))
    if not candidates:
        candidates = list(Path(results_dir).glob("**/ensembles/frd/selection_manifest.json"))
    # Prefer the manifest explicitly generated by the sentiment extension.
    candidates.sort(key=lambda p: ("sentiment_extension" not in str(p), len(str(p))))
    return candidates[0] if candidates else None


# -----------------------------------------------------------------------------
# Model definitions/loaders and inference
# -----------------------------------------------------------------------------


def _hybrid_classes():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class AttrDict(dict):
        __getattr__ = dict.__getitem__

    class SentimentFusionClassifier(nn.Module):
        def __init__(self, encoder: nn.Module, sentiment_dim: int, gated: bool = False, dropout: float = 0.15):
            super().__init__()
            self.encoder = encoder
            self.sentiment_dim = int(sentiment_dim)
            self.gated = bool(gated)
            hidden = int(getattr(encoder.config, "hidden_size"))
            self.text_norm = nn.LayerNorm(hidden)
            sent_hidden = min(128, max(16, hidden // 8))
            self.sentiment_proj = nn.Sequential(nn.Linear(sentiment_dim, sent_hidden), nn.GELU(), nn.Dropout(dropout))
            self.dropout = nn.Dropout(dropout)
            if gated:
                self.sentiment_to_hidden = nn.Linear(sent_hidden, hidden)
                self.gate = nn.Linear(hidden + sent_hidden, hidden)
                self.fusion_norm = nn.LayerNorm(hidden)
                self.classifier = nn.Linear(hidden, 2)
            else:
                self.classifier = nn.Linear(hidden + sent_hidden, 2)

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, sentiment_features=None, labels=None, **kwargs):
            enc_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                enc_kwargs["token_type_ids"] = token_type_ids
            outputs = self.encoder(**enc_kwargs)
            pooled = self.text_norm(outputs.last_hidden_state[:, 0])
            sent = self.sentiment_proj(sentiment_features.float())
            if self.gated:
                sent_h = self.sentiment_to_hidden(sent)
                gate = torch.sigmoid(self.gate(torch.cat([pooled, sent], dim=-1)))
                fused = self.fusion_norm(pooled + gate * sent_h)
            else:
                fused = torch.cat([pooled, sent], dim=-1)
            logits = self.classifier(self.dropout(fused))
            loss = F.cross_entropy(logits, labels) if labels is not None else None
            return AttrDict(loss=loss, logits=logits)

    return AttrDict, SentimentFusionClassifier


def load_plain_classifier(bundle: ModelBundle, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification

    meta = bundle.artifact_meta
    tok_dir = bundle.artifact / "tokenizer"
    tok = _load_artifact_tokenizer(tok_dir, bundle.key)
    adaptation = str(meta.get("adaptation", "full"))

    if adaptation == "peft":
        from peft import PeftModel
        hf_id = meta.get("hf_id")
        if not hf_id:
            raise RuntimeError(f"PEFT artifact for {bundle.key} does not contain hf_id")
        base = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=2)
        model = PeftModel.from_pretrained(base, bundle.artifact)
    elif _is_modernbert_key(bundle.key):
        # ModernBERT 4.48.x can be numerically unstable on some Windows/CUDA
        # combinations. Disable reference compilation and force FP32.
        cfg = _modernbert_config_from_artifact(bundle.artifact)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                bundle.artifact, config=cfg, torch_dtype=torch.float32,
                attn_implementation="eager"
            )
        except Exception:
            model = AutoModelForSequenceClassification.from_pretrained(
                bundle.artifact, config=cfg, torch_dtype=torch.float32
            )
        model = model.float()
    elif _is_deberta_v3_key(bundle.key):
        # Keep DeBERTa-v3 in FP32 for external inference. This also avoids
        # half-precision attention-mask overflows seen in the Windows setup.
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                bundle.artifact, torch_dtype=torch.float32,
                attn_implementation="eager"
            )
        except Exception:
            model = AutoModelForSequenceClassification.from_pretrained(
                bundle.artifact, torch_dtype=torch.float32
            )
        model = model.float()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(bundle.artifact)

    return model.to(device).eval(), tok


def load_hybrid_classifier(bundle: ModelBundle, device: str):
    import torch
    from transformers import AutoModel

    _, SentimentFusionClassifier = _hybrid_classes()
    meta = bundle.artifact_meta
    tok = _load_artifact_tokenizer(bundle.artifact / "tokenizer", bundle.key)

    encoder_path = bundle.artifact / "encoder"
    if _is_modernbert_key(bundle.key):
        cfg = _modernbert_config_from_artifact(encoder_path)
        try:
            encoder = AutoModel.from_pretrained(
                encoder_path, config=cfg, torch_dtype=torch.float32,
                attn_implementation="eager"
            )
        except Exception:
            encoder = AutoModel.from_pretrained(
                encoder_path, config=cfg, torch_dtype=torch.float32
            )
        encoder = encoder.float()
    elif _is_deberta_v3_key(bundle.key):
        try:
            encoder = AutoModel.from_pretrained(
                encoder_path, torch_dtype=torch.float32,
                attn_implementation="eager"
            )
        except Exception:
            encoder = AutoModel.from_pretrained(encoder_path, torch_dtype=torch.float32)
        encoder = encoder.float()
    else:
        encoder = AutoModel.from_pretrained(encoder_path)

    model = SentimentFusionClassifier(
        encoder,
        int(meta["sentiment_dim"]),
        gated=bool(meta.get("gated", False)),
    )
    state_path = bundle.artifact / "hybrid_state.pt"
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    if _is_modernbert_key(bundle.key) or _is_deberta_v3_key(bundle.key):
        model = model.float()
    return model.to(device).eval(), tok


def prepare_hybrid_features(bundle: ModelBundle, raw_features):
    import numpy as np

    meta = bundle.artifact_meta
    names = list(meta.get("feature_names", []))
    if not names:
        raise RuntimeError(f"Hybrid artifact {bundle.key} has no feature_names")
    unknown = [n for n in names if n not in FEATURE_NAMES]
    if unknown:
        raise RuntimeError(f"Hybrid artifact {bundle.key} contains unknown sentiment features: {unknown}")
    feats = raw_features[names].to_numpy(dtype=np.float32)

    # The FRD-fitted scaler is serialized into the artifact. It is applied as-is.
    # Nothing is fit/re-estimated from AiGen data.
    mean_full = np.asarray(meta.get("scaler_mean", [0.0] * len(FEATURE_NAMES)), dtype=np.float32)
    scale_full = np.asarray(meta.get("scaler_scale", [1.0] * len(FEATURE_NAMES)), dtype=np.float32)
    idx = [list(FEATURE_NAMES).index(n) for n in names]
    if len(mean_full) == len(FEATURE_NAMES):
        mean = mean_full[idx]
        scale = scale_full[idx]
    elif len(mean_full) == len(names):
        mean = mean_full
        scale = scale_full
    else:
        raise RuntimeError(
            f"Unexpected scaler dimension in {bundle.key}: mean={len(mean_full)}, features={len(names)}"
        )
    scale = np.where(scale == 0, 1.0, scale)
    return (feats - mean) / scale


def predict_bundle(
    bundle: ModelBundle,
    texts: list[str],
    raw_features,
    device: str,
    max_length: int,
    plain_batch: int,
    hybrid_batch: int,
    modernbert_device: str = "cpu",
):
    import numpy as np
    import torch

    is_hybrid = bundle.artifact_meta.get("type") == "hybrid"
    requested_device = device
    if _is_modernbert_key(bundle.key):
        if modernbert_device == "cpu":
            requested_device = "cpu"
        elif modernbert_device == "cuda":
            requested_device = "cuda"
        # auto -> ordinary selected device

    batch_initial = hybrid_batch if is_hybrid else plain_batch
    if _is_modernbert_key(bundle.key) and requested_device == "cpu":
        batch_initial = min(batch_initial, 4)
    if _is_deberta_v3_key(bundle.key):
        batch_initial = min(batch_initial, 8)

    def _run(run_device: str, initial_bs: int):
        last_exc: Exception | None = None
        for bs in batch_size_fallbacks(initial_bs):
            model = tok = None
            try:
                _cleanup_cuda()
                if is_hybrid:
                    model, tok = load_hybrid_classifier(bundle, run_device)
                    scaled = prepare_hybrid_features(bundle, raw_features)
                    if not np.isfinite(scaled).all():
                        bad = int(np.size(scaled) - np.isfinite(scaled).sum())
                        raise FloatingPointError(f"Non-finite scaled sentiment features: {bad}")
                else:
                    model, tok = load_plain_classifier(bundle, run_device)
                    scaled = None

                probs = []
                start = time.perf_counter()
                with torch.inference_mode():
                    for i in range(0, len(texts), bs):
                        batch_text = [str(x) for x in texts[i:i + bs]]
                        encoded = tok(
                            batch_text,
                            truncation=True,
                            max_length=max_length,
                            padding=True,
                            # Avoid shape-specific padding kernels for the
                            # conservative ModernBERT fallback path.
                            pad_to_multiple_of=None if _is_modernbert_key(bundle.key) else 8,
                            return_tensors="pt",
                        )
                        encoded = {k: v.to(run_device) for k, v in encoded.items()}
                        if is_hybrid:
                            encoded["sentiment_features"] = torch.tensor(
                                scaled[i:i + bs], dtype=torch.float32, device=run_device
                            )
                        logits = model(**encoded).logits
                        if not torch.isfinite(logits).all():
                            bad = int((~torch.isfinite(logits)).sum().detach().cpu().item())
                            raise FloatingPointError(
                                f"Non-finite logits in {bundle.key} at rows {i}:{min(i+bs, len(texts))}; "
                                f"count={bad}; device={run_device}; batch={bs}"
                            )
                        batch_prob = torch.softmax(logits.float(), dim=-1)[:, 1]
                        if not torch.isfinite(batch_prob).all():
                            raise FloatingPointError(
                                f"Non-finite probabilities in {bundle.key} at rows "
                                f"{i}:{min(i+bs, len(texts))}; device={run_device}; batch={bs}"
                            )
                        probs.append(batch_prob.detach().cpu().numpy())
                seconds = time.perf_counter() - start
                p = np.concatenate(probs).astype(float)
                if not np.isfinite(p).all():
                    raise FloatingPointError(f"Non-finite final probabilities for {bundle.key}")
                del model, tok
                _cleanup_cuda()
                return p, {
                    "seconds": float(seconds),
                    "reviews_per_second": float(len(texts) / seconds if seconds else 0.0),
                    "batch_size": int(bs),
                    "device": run_device,
                    "artifact_type": "hybrid" if is_hybrid else "classifier",
                    "modernbert_safe_mode": bool(_is_modernbert_key(bundle.key)),
                }
            except Exception as exc:
                last_exc = exc
                message = str(exc).lower()
                is_oom = "out of memory" in message
                try:
                    del model, tok
                except Exception:
                    pass
                _cleanup_cuda()
                if not is_oom or bs == 1:
                    raise
                print(f"[OOM] {bundle.key}: batch {bs} failed; retrying with smaller batch")
        assert last_exc is not None
        raise last_exc

    # Main attempt.  ModernBERT defaults to CPU safe mode because the source
    # checkpoints are already trained and external evaluation values numerical
    # correctness over throughput.
    try:
        return _run(requested_device, batch_initial)
    except FloatingPointError as exc:
        if _is_modernbert_key(bundle.key) and requested_device != "cpu":
            print(f"[WARN] {bundle.key}: {exc}")
            print("[WARN] Retrying ModernBERT in CPU/FP32 eager safe mode...")
            return _run("cpu", min(batch_initial, 4))
        raise


# -----------------------------------------------------------------------------
# Metrics and uncertainty
# -----------------------------------------------------------------------------


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    import numpy as np
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

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= float(threshold)).astype(int)
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


def bootstrap_ci(y_true, y_prob, metric: str, threshold: float, n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    import numpy as np

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = len(y)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        try:
            vals.append(float(compute_metrics(y[idx], p[idx], threshold)[metric]))
        except ValueError:
            continue
    if not vals:
        return {"low": float("nan"), "high": float("nan"), "n": 0}
    arr = np.asarray(vals, dtype=float)
    return {
        "low": float(np.quantile(arr, alpha / 2)),
        "high": float(np.quantile(arr, 1 - alpha / 2)),
        "n": int(len(arr)),
    }


def wilson_accuracy_ci(y_true, y_prob, threshold: float, alpha: float = 0.05) -> dict:
    """Wilson interval for overall accuracy (correct/incorrect Bernoulli trials)."""
    import numpy as np
    from scipy.stats import norm

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= float(threshold)).astype(int)
    n = len(y)
    k = int((pred == y).sum())
    phat = k / n
    z = float(norm.ppf(1 - alpha / 2))
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return {"low": float(center - half), "high": float(center + half), "n": int(n)}


def save_prediction_frame(path: Path, frame, prob, threshold: float, system_key: str, display_name: str, seed: int) -> None:
    import numpy as np
    import pandas as pd

    p = np.asarray(prob, dtype=float)
    pred = (p >= float(threshold)).astype(int)
    out = pd.DataFrame({
        "sample_id": frame["source_index"].to_numpy(),
        "y_true": frame["label_num"].to_numpy(dtype=int),
        "y_prob_machine_generated": p,
        "y_pred": pred,
        "threshold": float(threshold),
        "system_key": system_key,
        "system": display_name,
        "seed": int(seed),
    })
    _csv_atomic(out, path)


def system_metric_rows(system_key: str, display_name: str, system_type: str, threshold: float, y, prob, inference: dict | None = None) -> tuple[dict, dict]:
    primary = compute_metrics(y, prob, threshold)
    default = compute_metrics(y, prob, 0.5)
    common = {
        "system_key": system_key,
        "system": display_name,
        "system_type": system_type,
        "source_threshold": float(threshold),
    }
    if inference:
        common.update({
            "inference_seconds": inference.get("seconds"),
            "reviews_per_second": inference.get("reviews_per_second"),
            "inference_batch_size": inference.get("batch_size"),
        })
    return ({**common, **primary}, {**common, **default})


# -----------------------------------------------------------------------------
# Ensemble transfer: FRD selection/weights/thresholds are frozen
# -----------------------------------------------------------------------------


def _resolve_ensemble_source_dir(manifest_path: Path, selected: dict, method: str) -> Path:
    key = f"{method}_dir"
    value = selected.get(key)
    if value:
        p = Path(str(value))
        if p.is_dir() and (p / "ensemble_meta.json").is_file():
            return p
    rank = int(selected["rank"])
    members = tuple(selected["members"])
    group = f"triple_rank_{rank:02d}__" + "__".join(members)
    p = manifest_path.parent / group / method
    if p.is_dir() and (p / "ensemble_meta.json").is_file():
        return p
    raise FileNotFoundError(f"Could not resolve FRD ensemble metadata for rank {rank}, method {method}")


def evaluate_ensembles(
    manifest_path: Path,
    external_probs: dict[str, Any],
    frame,
    out_root: Path,
    seed: int,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    import numpy as np

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    primary_rows: list[dict] = []
    default_rows: list[dict] = []
    generated: dict[str, Any] = {}
    y = frame["label_num"].to_numpy(dtype=int)

    for selected in manifest.get("selected", []):
        rank = int(selected["rank"])
        members = tuple(selected["members"])
        missing = [m for m in members if m not in external_probs]
        if missing:
            print(f"[WARN] Skipping ensemble rank {rank}: missing external probabilities for {missing}")
            continue
        for method in ("equal_soft", "weighted_soft"):
            source_dir = _resolve_ensemble_source_dir(Path(manifest_path), selected, method)
            meta = json.loads((source_dir / "ensemble_meta.json").read_text(encoding="utf-8"))
            weights = {str(k): float(v) for k, v in meta.get("weights", {}).items()}
            if not weights:
                weights = {m: 1.0 / len(members) for m in members}
            wsum = sum(weights[m] for m in members)
            if wsum <= 0:
                raise RuntimeError(f"Non-positive ensemble weight sum for rank {rank} {method}")
            prob = sum(np.asarray(external_probs[m], dtype=float) * weights[m] for m in members) / wsum
            threshold = float(meta.get("selected_threshold", 0.5))
            group = f"triple_rank_{rank:02d}__" + "__".join(members)
            system_key = f"ensemble::{group}::{method}"
            display_name = f"E{rank} {method.replace('_', ' ').title()}"
            target = Path(out_root) / "ensembles" / group / method
            target.mkdir(parents=True, exist_ok=True)

            primary, default = system_metric_rows(
                system_key, display_name, "ensemble", threshold, y, prob,
                inference={"seconds": 0.0, "reviews_per_second": None, "batch_size": None},
            )
            primary.update({
                "ensemble_rank": rank,
                "members": " | ".join(members),
                "weights": json.dumps(weights, sort_keys=True),
                "selection_data": "FRD_validation_only",
            })
            default.update({
                "ensemble_rank": rank,
                "members": " | ".join(members),
                "weights": json.dumps(weights, sort_keys=True),
                "selection_data": "FRD_validation_only",
            })
            payload = {
                "protocol": "zero_shot_frd_to_aigen",
                "dataset": "AiGen-FoodReview official test split",
                "system_key": system_key,
                "system": display_name,
                "seed": seed,
                "members": members,
                "weights": weights,
                "source_threshold": threshold,
                "selection_data": "FRD_validation_only",
                "primary_frd_validation_threshold": primary,
                "secondary_default_0_5": default,
            }
            _json_dump(target / "metrics.json", payload)
            save_prediction_frame(target / "predictions.csv", frame, prob, threshold, system_key, display_name, seed)
            primary_rows.append(primary)
            default_rows.append(default)
            generated[system_key] = {
                "prob": prob,
                "threshold": threshold,
                "display_name": display_name,
                "rank": rank,
                "method": method,
                "members": members,
                "weights": weights,
                "metrics_path": str((target / "metrics.json").resolve()),
                "predictions_path": str((target / "predictions.csv").resolve()),
            }
    return primary_rows, default_rows, generated


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def add_confidence_intervals(rows: list[dict], probs: dict[str, Any], y, n_boot: int, seed: int):
    import pandas as pd

    ci_rows: list[dict] = []
    by_key = {r["system_key"]: r for r in rows}
    for system_key, r in by_key.items():
        if system_key not in probs:
            continue
        p = probs[system_key]
        th = float(r["source_threshold"])
        for metric in ("accuracy", "f1", "roc_auc"):
            ci = bootstrap_ci(y, p, metric, th, n_boot=n_boot, seed=seed)
            ci_rows.append({
                "system_key": system_key,
                "system": r["system"],
                "metric": metric,
                "point": float(r[metric]),
                "ci_low": ci["low"],
                "ci_high": ci["high"],
                "n_boot": ci["n"],
                "threshold": th,
            })
        wilson = wilson_accuracy_ci(y, p, th)
        ci_rows.append({
            "system_key": system_key,
            "system": r["system"],
            "metric": "accuracy_wilson",
            "point": float(r["accuracy"]),
            "ci_low": wilson["low"],
            "ci_high": wilson["high"],
            "n_boot": None,
            "threshold": th,
        })
    return pd.DataFrame(ci_rows)


def attach_primary_ci_columns(metrics_df, ci_df):
    out = metrics_df.copy()
    for metric in ("accuracy", "f1", "roc_auc"):
        sub = ci_df[ci_df.metric == metric][["system_key", "ci_low", "ci_high"]].rename(
            columns={"ci_low": f"{metric}_ci_low", "ci_high": f"{metric}_ci_high"}
        )
        out = out.merge(sub, on="system_key", how="left")
    wil = ci_df[ci_df.metric == "accuracy_wilson"][["system_key", "ci_low", "ci_high"]].rename(
        columns={"ci_low": "accuracy_wilson_ci_low", "ci_high": "accuracy_wilson_ci_high"}
    )
    return out.merge(wil, on="system_key", how="left")


def paper_table(metrics_df, representative_info: dict, final_key: str | None):
    selected = set(representative_info.get("representatives", []))
    wanted = [f"model::{k}" for k in representative_info.get("representatives", [])]
    if final_key:
        wanted.append(final_key)
    rows = []
    lookup = {str(r.system_key): r for _, r in metrics_df.iterrows()}
    for key in wanted:
        if key in lookup:
            rows.append(lookup[key].to_dict())
    import pandas as pd
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = [c for c in [
        "system", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
        "precision", "recall", "f1", "f1_ci_low", "f1_ci_high",
        "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc",
        "mcc", "specificity", "balanced_accuracy", "tn", "fp", "fn", "tp",
        "source_threshold", "system_key",
    ] if c in df.columns]
    return df[cols]


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Zero-shot FRD -> AiGen-FoodReview external validation")
    p.add_argument("--dataset", default="test.csv", help="AiGen-FoodReview official test.csv")
    p.add_argument("--results-dir", default="results", help="Completed FRD results directory")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", default=None, help="Default: results/external_aigen/seed_<seed>")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--plain-batch", type=int, default=16)
    p.add_argument("--hybrid-batch", type=int, default=8)
    p.add_argument("--modernbert-device", choices=["auto", "cuda", "cpu"], default="cpu",
                   help="ModernBERT inference device. Default cpu = conservative FP32/eager fallback for Windows stability.")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    p.add_argument("--scope", choices=["all", "selected", "final"], default="all",
                   help="all=14 models + ensembles; selected=7 FRD-selected reps + ensembles; final=final ensemble members + final ensemble")
    p.add_argument("--strict", action="store_true", help="Fail if any requested FRD model artifact is missing")
    p.add_argument("--force", action="store_true", help="Recompute predictions even if compatible cached predictions exist")
    p.add_argument("--dry-run", action="store_true", help="Audit dataset/artifacts only; no model inference")
    return p.parse_args(argv)


def _requested_model_keys(scope: str, representative_info: dict, manifest: dict | None) -> list[str]:
    if scope == "all":
        return list(MODEL_DISPLAY_NAMES)
    if scope == "selected":
        return list(representative_info.get("representatives", []))
    # final: members of rank-1 selected triple, falling back to selected reps.
    if manifest:
        selected = sorted(manifest.get("selected", []), key=lambda x: int(x.get("rank", 999)))
        if selected:
            return list(selected[0].get("members", []))
    return list(representative_info.get("representatives", []))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _patch_torch_compile_for_windows()
    project = Path(__file__).resolve().parent
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = project / dataset_path
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project / results_dir
    out_root = Path(args.output_dir) if args.output_dir else results_dir / "external_aigen" / f"seed_{args.seed}"
    if not out_root.is_absolute():
        out_root = project / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("FRD -> AiGen-FoodReview ZERO-SHOT EXTERNAL VALIDATION")
    print("Runner:", RUNNER_VERSION)
    print("Project:", project)
    print("Dataset:", dataset_path)
    print("FRD results:", results_dir)
    print("Output:", out_root)
    print("Seed:", args.seed)
    print("Scope:", args.scope)
    print("=" * 78)

    raw_df, frame = load_aigen_test(dataset_path)
    audit = dataset_audit(dataset_path, raw_df, frame, results_dir)
    _json_dump(out_root / "dataset_audit.json", audit)
    print("\n[DATASET]")
    print(f"Rows: {audit['rows']:,}")
    print("Class 0 authentic:", audit["class_counts"]["0_authentic"])
    print("Class 1 machine-generated:", audit["class_counts"]["1_machine_generated"])
    print("MD5:", audit["md5"])
    print("Official AiGen test.csv MD5 match:", audit["official_test_md5_match"])
    if audit.get("frd_overlap"):
        print("FRD normalized exact overlap audit:", audit["frd_overlap"])

    bundles, missing = discover_all_bundles(results_dir, args.seed)
    representative_info = find_representative_selection(results_dir, args.seed, bundles)
    manifest_path = find_ensemble_manifest(results_dir, args.seed)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else None
    requested = _requested_model_keys(args.scope, representative_info, manifest)

    print("\n[FRD ARTIFACTS]")
    print("Available model artifacts:", len(bundles), "/", len(MODEL_DISPLAY_NAMES))
    for key in requested:
        if key in bundles:
            b = bundles[key]
            print(f"  OK  {key:34s} threshold={b.source_threshold:.6f}  {b.artifact}")
        else:
            print(f"  MISS {key:34s} {missing.get(key, 'not found')}")
    print("Representative selection:", representative_info.get("selected_by_backbone", {}))
    print("Ensemble manifest:", manifest_path if manifest_path else "MISSING")

    missing_requested = [k for k in requested if k not in bundles]
    audit_payload = {
        "requested_models": requested,
        "available_models": sorted(bundles),
        "missing_requested_models": missing_requested,
        "missing_all_models": missing,
        "representative_selection": representative_info,
        "ensemble_manifest": str(manifest_path.resolve()) if manifest_path else None,
    }
    _json_dump(out_root / "artifact_audit.json", audit_payload)

    if args.strict and missing_requested:
        raise RuntimeError(f"Missing requested FRD artifacts: {missing_requested}")
    if args.dry_run:
        print("\n[DRY RUN] No inference performed.")
        return 0
    if not any(k in bundles for k in requested):
        raise RuntimeError("No requested FRD model artifacts are available; nothing to evaluate")

    device = _device_from_arg(args.device)
    print("\n[DEVICE]", device)
    if device == "cpu":
        print("[WARN] CPU inference is supported but will be much slower than CUDA.")

    dataset_sha256 = audit["sha256"]
    raw_features = build_sentiment_features(frame, out_root / "aigen_sentiment_features.csv", dataset_sha256)
    texts = frame["text"].astype(str).tolist()
    y = frame["label_num"].to_numpy(dtype=int)

    primary_rows: list[dict] = []
    default_rows: list[dict] = []
    external_model_probs: dict[str, Any] = {}
    all_probs_for_ci: dict[str, Any] = {}
    failures: dict[str, str] = {}

    print("\n" + "=" * 78)
    print("MODEL INFERENCE -- NO TRAINING / NO AIGEN THRESHOLD SELECTION")
    print("=" * 78)

    for idx, key in enumerate(requested, 1):
        if key not in bundles:
            continue
        bundle = bundles[key]
        display = MODEL_DISPLAY_NAMES.get(key, key)
        target = out_root / "models" / key
        target.mkdir(parents=True, exist_ok=True)
        pred_path = target / "predictions.csv"
        metrics_path = target / "metrics.json"
        try:
            prob = None
            perf = None
            if not args.force and pred_path.is_file() and metrics_path.is_file():
                import pandas as pd
                cached = pd.read_csv(pred_path)
                if (
                    len(cached) == len(frame)
                    and "sample_id" in cached.columns
                    and "y_prob_machine_generated" in cached.columns
                    and cached["sample_id"].astype(str).tolist() == frame["source_index"].astype(str).tolist()
                ):
                    prob = cached["y_prob_machine_generated"].to_numpy(dtype=float)
                    old = json.loads(metrics_path.read_text(encoding="utf-8"))
                    perf = old.get("inference", {"cached": True})
                    print(f"[{idx:02d}/{len(requested):02d}] CACHE {display}")
            if prob is None:
                print(f"[{idx:02d}/{len(requested):02d}] RUN   {display}")
                prob, perf = predict_bundle(
                    bundle, texts, raw_features, device, args.max_length,
                    args.plain_batch, args.hybrid_batch,
                    modernbert_device=args.modernbert_device,
                )

            threshold = bundle.source_threshold
            primary, default = system_metric_rows(
                f"model::{key}", display, "model", threshold, y, prob, inference=perf,
            )
            payload = {
                "protocol": "zero_shot_frd_to_aigen",
                "dataset": "AiGen-FoodReview official test split",
                "dataset_sha256": dataset_sha256,
                "model_key": key,
                "system": display,
                "seed": args.seed,
                "artifact": str(bundle.artifact.resolve()),
                "artifact_type": bundle.artifact_meta.get("type", "classifier"),
                "source_threshold": threshold,
                "threshold_source": "FRD_validation_only",
                "sentiment_scaler_source": "FRD_training_only (stored in artifact)" if bundle.artifact_meta.get("type") == "hybrid" else None,
                "primary_frd_validation_threshold": primary,
                "secondary_default_0_5": default,
                "inference": perf,
            }
            _json_dump(metrics_path, payload)
            save_prediction_frame(pred_path, frame, prob, threshold, f"model::{key}", display, args.seed)
            # A previous failed attempt may have left error.json behind. Once
            # compatible predictions/metrics are successfully produced, that
            # stale error marker is no longer authoritative.
            stale_error = target / "error.json"
            if stale_error.is_file():
                try:
                    stale_error.unlink()
                except OSError:
                    pass
            primary_rows.append(primary)
            default_rows.append(default)
            external_model_probs[key] = prob
            all_probs_for_ci[f"model::{key}"] = prob
            print(
                f"       Acc={primary['accuracy']:.4f} F1={primary['f1']:.4f} "
                f"AUC={primary['roc_auc']:.4f}  threshold={threshold:.4f}"
            )
        except Exception as exc:
            failures[key] = traceback.format_exc()
            _json_dump(target / "error.json", {"error": repr(exc), "traceback": failures[key]})
            print(f"[ERROR] {display}: {exc}", file=sys.stderr)
            _cleanup_cuda()
            if args.strict:
                raise

    # Transfer the FRD validation-selected ensembles unchanged.
    ensemble_generated: dict[str, Any] = {}
    if manifest_path and external_model_probs:
        try:
            ens_primary, ens_default, ensemble_generated = evaluate_ensembles(
                manifest_path, external_model_probs, frame, out_root, args.seed
            )
            # final scope means only rank-1 weighted result is reported as ensemble;
            # other generated ensemble files are harmless but we filter report rows.
            if args.scope == "final":
                ens_primary = [r for r in ens_primary if r.get("ensemble_rank") == 1 and r["system_key"].endswith("::weighted_soft")]
                ens_default = [r for r in ens_default if r.get("ensemble_rank") == 1 and r["system_key"].endswith("::weighted_soft")]
            primary_rows.extend(ens_primary)
            default_rows.extend(ens_default)
            for system_key, info in ensemble_generated.items():
                if args.scope == "final" and not (info.get("rank") == 1 and info.get("method") == "weighted_soft"):
                    continue
                all_probs_for_ci[system_key] = info["prob"]
        except Exception as exc:
            failures["ensembles"] = traceback.format_exc()
            _json_dump(out_root / "ensembles_error.json", {"error": repr(exc), "traceback": failures["ensembles"]})
            print(f"[ERROR] Ensemble transfer failed: {exc}", file=sys.stderr)
            if args.strict:
                raise
    else:
        print("[WARN] No FRD sentiment-extension ensemble manifest found; model-level external results only.")

    import pandas as pd
    primary_df = pd.DataFrame(primary_rows)
    default_df = pd.DataFrame(default_rows)
    if primary_df.empty:
        raise RuntimeError("No external metrics were produced")

    # Stable report order: models in study order, then ensemble rank/method.
    model_order = {f"model::{k}": i for i, k in enumerate(MODEL_DISPLAY_NAMES)}
    primary_df["_order"] = primary_df["system_key"].map(model_order).fillna(10_000)
    primary_df = primary_df.sort_values(["_order", "system_key"]).drop(columns="_order").reset_index(drop=True)
    default_df["_order"] = default_df["system_key"].map(model_order).fillna(10_000)
    default_df = default_df.sort_values(["_order", "system_key"]).drop(columns="_order").reset_index(drop=True)

    _csv_atomic(primary_df, out_root / "all_systems_primary_metrics.csv")
    _csv_atomic(default_df, out_root / "all_systems_default_05_metrics.csv")

    print(f"\n[BOOTSTRAP] {args.bootstrap} resamples/system for Accuracy, F1, ROC-AUC...")
    ci_df = add_confidence_intervals(
        primary_df.to_dict("records"), all_probs_for_ci, y,
        n_boot=args.bootstrap, seed=args.bootstrap_seed,
    )
    _csv_atomic(ci_df, out_root / "confidence_intervals.csv")
    primary_with_ci = attach_primary_ci_columns(primary_df, ci_df)
    _csv_atomic(primary_with_ci, out_root / "all_systems_primary_metrics_with_ci.csv")

    # Final system is the FRD validation-selected rank-1 weighted soft-voting ensemble.
    final_key = None
    if ensemble_generated:
        candidates = [
            (k, v) for k, v in ensemble_generated.items()
            if v.get("rank") == 1 and v.get("method") == "weighted_soft"
        ]
        if candidates:
            final_key = candidates[0][0]

    paper_df = paper_table(primary_with_ci, representative_info, final_key)
    _csv_atomic(paper_df, out_root / "paper_table_selected_representatives_plus_final.csv")

    final_payload = None
    if final_key:
        final_row = primary_with_ci[primary_with_ci.system_key == final_key]
        if not final_row.empty:
            info = ensemble_generated[final_key]
            final_payload = {
                "system_key": final_key,
                "system": final_row.iloc[0].to_dict(),
                "members": list(info["members"]),
                "weights": info["weights"],
                "threshold": info["threshold"],
                "selection_data": "FRD_validation_only",
                "external_dataset_used_for_selection": False,
            }
            _json_dump(out_root / "final_system.json", final_payload)

    summary = {
        "runner_version": RUNNER_VERSION,
        "status": "COMPLETE" if not failures and not missing_requested else "COMPLETE_WITH_WARNINGS",
        "protocol": "FRD-trained -> AiGen-FoodReview official test split, zero-shot",
        "seed": args.seed,
        "scope": args.scope,
        "device": device,
        "dataset": audit,
        "requested_models": requested,
        "evaluated_model_keys": [r["system_key"] for r in primary_rows if r.get("system_type") == "model"],
        "representative_selection": representative_info,
        "ensemble_manifest": str(manifest_path.resolve()) if manifest_path else None,
        "final_system_key": final_key,
        "missing_requested_models": missing_requested,
        "failures": {k: v.splitlines()[-1] if v else "" for k, v in failures.items()},
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "no_aigen_training": True,
        "no_aigen_threshold_selection": True,
        "no_aigen_ensemble_selection": True,
        "no_aigen_sentiment_scaler_fitting": True,
        "output_root": str(out_root.resolve()),
    }
    _json_dump(out_root / "run_summary.json", summary)

    print("\n" + "=" * 78)
    print("EXTERNAL VALIDATION COMPLETE")
    print("=" * 78)
    print("Primary metrics:", out_root / "all_systems_primary_metrics_with_ci.csv")
    print("Paper table:", out_root / "paper_table_selected_representatives_plus_final.csv")
    if final_payload:
        r = final_payload["system"]
        print("Final FRD-selected ensemble on AiGen:")
        print("  Members:", ", ".join(final_payload["members"]))
        print("  Weights:", final_payload["weights"])
        print("  Frozen threshold:", final_payload["threshold"])
        print(f"  Accuracy={r['accuracy']:.4f}  F1={r['f1']:.4f}  ROC-AUC={r['roc_auc']:.4f}")
    if missing_requested:
        print("WARNING - missing requested models:", missing_requested)
    if failures:
        print("WARNING - failed systems:", list(failures))
    return 0 if not failures else 2


# =============================================================================
# Post-hoc checks (read stored predictions; no training, no selection)
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent


class _ThisModule:
    """Lets the post-hoc checks call the evaluation functions above as EXT.<name>."""
    def __getattr__(self, name):
        return globals()[name]


EXT = _ThisModule()
MODEL_DISPLAY_NAMES = MODEL_DISPLAY_NAMES
BACKBONE_CANDIDATES = BACKBONE_CANDIDATES


# ---- diagnostics: truncation, surface features, stratified ROC-AUC
# -*- coding: utf-8 -*-



TITLE_RE = re.compile(r"^\s*title\s*:", re.IGNORECASE)
MIN_PER_CLASS = 30  # minimum class size for a stratum-level AUC


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"ID", "text", "label"} - set(df.columns)
    if missing:
        raise RuntimeError(f"{path} lacks columns {sorted(missing)}")
    out = pd.DataFrame({
        "sample_id": df["ID"].astype(str).to_numpy(),
        "text": df["text"].astype(str).to_numpy(),
        "y": pd.to_numeric(df["label"]).astype(int).to_numpy(),
    })
    if set(out["y"].unique()) != {0, 1}:
        raise RuntimeError("Expected labels {0,1} (0 = authentic, 1 = machine-generated)")
    return out


def _aligned_prob(pred_path: Path, test: pd.DataFrame) -> tuple[np.ndarray, float, str, str]:
    pr = pd.read_csv(pred_path)
    pr["sample_id"] = pr["sample_id"].astype(str)
    merged = test[["sample_id", "y"]].merge(pr, on="sample_id", how="left", validate="one_to_one")
    if merged["y_prob_machine_generated"].isna().any():
        raise RuntimeError(f"{pred_path}: predictions do not cover every test ID")
    if (merged["y"].to_numpy() != merged["y_true"].to_numpy()).any():
        raise RuntimeError(f"{pred_path}: stored labels disagree with test.csv")
    p = merged["y_prob_machine_generated"].to_numpy(dtype=float)
    if not np.isfinite(p).all():
        raise RuntimeError(f"{pred_path}: non-finite probabilities present")
    return p, float(merged["threshold"].iloc[0]), str(merged["system"].iloc[0]), str(merged["system_key"].iloc[0])


def load_systems(ext_dir: Path, test: pd.DataFrame) -> tuple["OrderedDict[str, dict]", str | None]:
    systems: "OrderedDict[str, dict]" = OrderedDict()
    for key, display in MODEL_DISPLAY_NAMES.items():
        path = ext_dir / "models" / key / "predictions.csv"
        if not path.is_file():
            print(f"[WARN] missing predictions for {display}: {path}")
            continue
        p, tau, _, skey = _aligned_prob(path, test)
        systems[skey] = {"name": display, "model_key": key, "prob": p, "tau": tau,
                         "kind": "model", "members": [key]}
    for path in sorted(ext_dir.glob("ensembles/*/*/predictions.csv")):
        p, tau, display, skey = _aligned_prob(path, test)
        group = path.parent.parent.name  # triple_rank_01__a__b__c
        members = group.split("__")[1:]
        systems[skey] = {"name": display, "model_key": None, "prob": p, "tau": tau,
                         "kind": "ensemble", "members": members}
    final_key = None
    fpath = ext_dir / "final_system.json"
    if fpath.is_file():
        final_key = json.loads(fpath.read_text(encoding="utf-8")).get("system_key")
    return systems, final_key


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def auc_boot(y: np.ndarray, p: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Point AUC and percentile CI; same resampling scheme as the original run."""
    point = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    v = np.asarray(vals)
    return point, float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def paired_auc_diff(y, pa, pb, n_boot, seed) -> tuple[float, float, float]:
    point = float(roc_auc_score(y, pa) - roc_auc_score(y, pb))
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    v = np.asarray(vals)
    return point, float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def op_point(y: np.ndarray, p: np.ndarray, tau: float) -> dict:
    pred = (p >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": (tp + tn) / len(y), "recall": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp), "flagged": tp + fp}


def chance_status(lo: float, hi: float) -> str:
    if lo > 0.5:
        return "above"
    if hi < 0.5:
        return "below"
    return "includes_0.5"


def subset_auc(y, p, mask, n_boot, seed) -> dict:
    ys, ps = y[mask], p[mask]
    n1, n0 = int((ys == 1).sum()), int((ys == 0).sum())
    if n1 < MIN_PER_CLASS or n0 < MIN_PER_CLASS:
        return {"auc": np.nan, "lo": np.nan, "hi": np.nan, "n_pos": n1, "n_neg": n0}
    a, lo, hi = auc_boot(ys, ps, n_boot, seed)
    return {"auc": a, "lo": lo, "hi": hi, "n_pos": n1, "n_neg": n0}


def length_stratified_auc(y, p, words, n_bins=5) -> dict:
    """Weighted mean of within-quintile AUCs (weights n_pos * n_neg)."""
    edges = np.unique(np.quantile(words, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, words, side="right") - 1, 0, len(edges) - 2)
    num = den = 0.0
    used = 0
    for b in np.unique(bins):
        m = bins == b
        n1, n0 = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if n1 < MIN_PER_CLASS or n0 < MIN_PER_CLASS:
            continue
        w = n1 * n0
        num += w * roc_auc_score(y[m], p[m]); den += w; used += 1
    return {"auc": num / den if den else np.nan, "strata_used": used}


# ----------------------------------------------------------------------------
# Tokenizer-based truncation (reuses the stored FRD tokenizers)
# ----------------------------------------------------------------------------

def token_lengths(results_dir: Path, seed: int, keys: list[str], texts: list[str]) -> dict[str, np.ndarray]:
    if False:  # EXT is this module
        return {}
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print(f"[WARN] transformers unavailable ({exc}); truncation step skipped")
        return {}
    out: dict[str, np.ndarray] = {}
    cache: dict[str, np.ndarray] = {}
    for key in keys:
        try:
            bundle = EXT.resolve_model_bundle(results_dir, seed, key)
        except Exception as exc:
            print(f"[WARN] artifact for {key} unavailable: {exc}")
            continue
        tok = None
        for use_fast in (True, False):  # DeBERTa-v3 artifacts store a slow SentencePiece tokenizer
            try:
                tok = AutoTokenizer.from_pretrained(bundle.artifact / "tokenizer", use_fast=use_fast)
                break
            except Exception as exc:
                last = exc
        if tok is None:
            print(f"[WARN] tokenizer for {key} unavailable: {last}")
            continue
        tok.model_max_length = 10 ** 9  # silence length warnings; nothing is truncated here
        sig = f"{tok.__class__.__name__}|{len(tok)}|{tok.convert_ids_to_tokens(tok('a b')['input_ids'])}"
        if sig not in cache:
            enc = tok(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
            cache[sig] = np.asarray([len(e) for e in enc])
        out[key] = cache[sig]
    return out


# ----------------------------------------------------------------------------
# Optional re-inference on format-normalized text
# ----------------------------------------------------------------------------

def normalize_format(t: str) -> str:
    t = TITLE_RE.sub("", t, count=1)
    return re.sub(r"\s+", " ", t).strip()


def reinfer(results_dir, seed, keys, test, stored, device_arg, batch, max_length, n_boot, bseed):
    rows = []
    if False:  # EXT is this module
        return rows
    try:
        import torch
    except Exception as exc:
        print(f"[WARN] torch unavailable ({exc}); re-inference skipped")
        return rows
    y = test["y"].to_numpy()
    texts_norm = [normalize_format(t) for t in test["text"]]
    for key in keys:
        try:
            bundle = EXT.resolve_model_bundle(results_dir, seed, key)
        except Exception as exc:
            print(f"[WARN] reinfer {key}: {exc}")
            continue
        if bundle.artifact_meta.get("type") == "hybrid":
            print(f"[SKIP] {key}: hybrid models need re-computed sentiment features; plain models only")
            continue
        device = EXT._device_from_arg(device_arg)

        def run(texts, dev, bs):
            p, info = EXT.predict_bundle(bundle, texts, None, dev, max_length, bs, bs)
            return np.asarray(p, dtype=float), info

        # Sanity check: re-scoring the ORIGINAL text must reproduce stored probabilities.
        n_chk = min(128, len(test))
        p_chk, info = run(list(test["text"].iloc[:n_chk]), device, batch)
        if not np.isfinite(p_chk).all() and device == "cuda":
            print(f"[INFO] {key}: non-finite GPU outputs; falling back to CPU, batch 4")
            device = "cpu"
            p_chk, info = run(list(test["text"].iloc[:n_chk]), device, 4)
        max_dev = float(np.max(np.abs(p_chk - stored[key][:n_chk])))
        p_norm, info = run(texts_norm, device, batch if device == "cuda" else 4)
        if not np.isfinite(p_norm).all():
            print(f"[WARN] {key}: non-finite outputs on normalized text")
            continue
        a0 = float(roc_auc_score(y, stored[key]))
        a1, lo, hi = auc_boot(y, p_norm, n_boot, bseed)
        rows.append({"model_key": key, "device": device,
                     "reproduce_check_n": n_chk, "reproduce_max_abs_diff": max_dev,
                     "auc_original_text": a0, "auc_format_normalized": a1,
                     "auc_norm_ci_low": lo, "auc_norm_ci_high": hi})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def diagnostics_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Post-hoc FRD->AiGen transfer diagnostics (no training, no selection)")
    ap.add_argument("--dataset", default="test.csv")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--external-dir", default=None)
    ap.add_argument("--frd-dataset", default=None, help="optional FRD CSV (column text_) for truncation comparison")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=314159)
    ap.add_argument("--reinfer", action="store_true")
    ap.add_argument("--reinfer-keys", default="bert,modernbert_base")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    ext_dir = Path(args.external_dir) if args.external_dir else results_dir / "external_aigen" / f"seed_{args.seed}"
    out_dir = ext_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    test = load_test(Path(args.dataset))
    y = test["y"].to_numpy()
    systems, final_key = load_systems(ext_dir, test)
    if not systems:
        print("No prediction files found."); return 2
    B, S = args.bootstrap, args.bootstrap_seed
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())

    # 1. Page-18 check --------------------------------------------------------
    rows = []
    for skey, s in systems.items():
        p = s["prob"]
        at_tau, at_05 = op_point(y, p, s["tau"]), op_point(y, p, 0.5)
        a, lo, hi = auc_boot(y, p, B, S)
        rows.append({
            "system": s["name"], "system_key": skey, "kind": s["kind"],
            "is_final": skey == final_key, "tau": s["tau"],
            **{k: at_tau[k] for k in ("tn", "fp", "fn", "tp", "flagged")},
            "accuracy": at_tau["accuracy"], "recall": at_tau["recall"], "specificity": at_tau["specificity"],
            "accuracy_at_0.5": at_05["accuracy"], "recall_at_0.5": at_05["recall"],
            "abs_acc_change_pp_tau_to_0.5": 100 * abs(at_05["accuracy"] - at_tau["accuracy"]),
            "median_p_authentic": float(np.median(p[y == 0])), "median_p_machine": float(np.median(p[y == 1])),
            "roc_auc": a, "auc_ci_low": lo, "auc_ci_high": hi, "auc_vs_chance": chance_status(lo, hi),
        })
    op_table = pd.DataFrame(rows)
    op_table.to_csv(out_dir / "operating_point_check.csv", index=False)

    ind = op_table[op_table.kind == "model"]
    summary = {
        "n_authentic": n_neg, "n_machine": n_pos,
        "all_authentic_accuracy": n_neg / len(y),
        "individual_systems": int(len(ind)),
        "accuracy_range": [float(ind.accuracy.min()), float(ind.accuracy.max())],
        "recall_max_at_tau": float(ind.recall.max()),
        "specificity_range": [float(ind.specificity.min()), float(ind.specificity.max())],
        "recall_max_at_0.5": float(ind["recall_at_0.5"].max()),
        "max_abs_acc_change_pp_tau_to_0.5": float(ind["abs_acc_change_pp_tau_to_0.5"].max()),
        "max_median_prob_any_class": float(ind[["median_p_authentic", "median_p_machine"]].to_numpy().max()),
        "auc_above_chance": ind.loc[ind.auc_vs_chance == "above", "system"].tolist(),
        "auc_below_chance": ind.loc[ind.auc_vs_chance == "below", "system"].tolist(),
        "auc_includes_0.5": ind.loc[ind.auc_vs_chance == "includes_0.5", "system"].tolist(),
    }

    # 2. Surface features -----------------------------------------------------
    texts = test["text"].tolist()
    words = np.asarray([len(re.findall(r"\S+", t)) for t in texts])
    has_title = np.asarray([bool(TITLE_RE.match(t)) for t in texts])
    has_break = np.asarray([("\n" in t) or ("\r" in t) for t in texts])
    surf = {
        "median_words": {"authentic": float(np.median(words[y == 0])), "machine": float(np.median(words[y == 1])),
                         "all": float(np.median(words))},
        "title_share": {"authentic": float(has_title[y == 0].mean()), "machine": float(has_title[y == 1].mean())},
        "linebreak_share": {"authentic": float(has_break[y == 0].mean()), "machine": float(has_break[y == 1].mean())},
        "auc_word_count": float(roc_auc_score(y, words)),
        "auc_title_indicator": float(roc_auc_score(y, has_title.astype(float))),
        "auc_linebreak_indicator": float(roc_auc_score(y, has_break.astype(float))),
        "auc_title_or_break_indicator": float(roc_auc_score(y, (has_title | has_break).astype(float))),
    }
    ff_mask = (y == 0) | ((y == 1) & ~has_title & ~has_break)
    surf["machine_without_title_or_break"] = int(((y == 1) & ~has_title & ~has_break).sum())

    # 3/4. Stratified AUC and truncation ---------------------------------------
    model_keys = [s["model_key"] for s in systems.values() if s["kind"] == "model"]
    lengths = token_lengths(results_dir, args.seed, model_keys, texts)
    trunc_rows = []
    for key, L in lengths.items():
        tr = L > args.max_length
        trunc_rows.append({"model_key": key, "aigen_truncated_all": float(tr.mean()),
                           "aigen_truncated_authentic": float(tr[y == 0].mean()),
                           "aigen_truncated_machine": float(tr[y == 1].mean())})
    if args.frd_dataset and lengths:
        frd_texts = pd.read_csv(args.frd_dataset)["text_"].astype(str).tolist()
        frd_len = token_lengths(results_dir, args.seed, list(lengths), frd_texts)
        for r in trunc_rows:
            if r["model_key"] in frd_len:
                r["frd_truncated_all"] = float((frd_len[r["model_key"]] > args.max_length).mean())
    pd.DataFrame(trunc_rows).to_csv(out_dir / "truncation.csv", index=False)

    strat_rows = []
    for skey, s in systems.items():
        p = s["prob"]
        ff = subset_auc(y, p, ff_mask, B, S)
        ls = length_stratified_auc(y, p, words)
        row = {"system": s["name"], "is_final": skey == final_key,
               "auc_all": float(roc_auc_score(y, p)),
               "auc_format_free": ff["auc"], "ff_ci_low": ff["lo"], "ff_ci_high": ff["hi"],
               "ff_n_machine": ff["n_pos"],
               "auc_length_stratified": ls["auc"], "length_strata_used": ls["strata_used"]}
        mem = [m for m in s["members"] if m in lengths]
        if mem and len(mem) == len(s["members"]):
            keep = np.all([lengths[m] <= args.max_length for m in mem], axis=0)
            nt = subset_auc(y, p, keep, B, S)
            row.update({"auc_not_truncated": nt["auc"], "nt_ci_low": nt["lo"], "nt_ci_high": nt["hi"],
                        "nt_n": int(keep.sum())})
        strat_rows.append(row)
    strat = pd.DataFrame(strat_rows)
    strat.to_csv(out_dir / "stratified_auc.csv", index=False)

    # 5. BERT vs final ensemble -------------------------------------------------
    bert_key = next((k for k, s in systems.items() if s["model_key"] == "bert"), None)
    if bert_key and final_key in systems:
        d, lo, hi = paired_auc_diff(y, systems[bert_key]["prob"], systems[final_key]["prob"], B, S)
        summary["auc_diff_bert_minus_final"] = {"point": d, "ci_low": lo, "ci_high": hi}

    # 6. Optional re-inference ---------------------------------------------------
    re_rows = []
    if args.reinfer:
        stored = {s["model_key"]: s["prob"] for s in systems.values() if s["kind"] == "model"}
        keys = [k.strip() for k in args.reinfer_keys.split(",") if k.strip() in stored]
        re_rows = reinfer(results_dir, args.seed, keys, test, stored, args.device,
                          args.batch, args.max_length, B, S)
        pd.DataFrame(re_rows).to_csv(out_dir / "reinfer_format_normalized.csv", index=False)

    summary["surface_features"] = surf
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    # Console report ------------------------------------------------------------------
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("\n=== 1. OPERATING-POINT CHECK (transferred thresholds) ===")
    print(op_table[["system", "tau", "tn", "fp", "fn", "tp", "accuracy", "recall", "specificity",
                  "recall_at_0.5", "abs_acc_change_pp_tau_to_0.5", "median_p_authentic",
                  "median_p_machine", "roc_auc", "auc_ci_low", "auc_ci_high", "auc_vs_chance"]]
          .round(4).to_string(index=False))
    print("\nSummary:", json.dumps({k: v for k, v in summary.items() if k != "surface_features"}, indent=1, default=float))
    print("\n=== 2. SURFACE FEATURES ===\n", json.dumps(surf, indent=1))
    print("\n=== 3. STRATIFIED AUC ===")
    print(strat.round(4).to_string(index=False))
    if trunc_rows:
        print("\n=== 4. TRUNCATION (share > max_length subword tokens) ===")
        print(pd.DataFrame(trunc_rows).round(4).to_string(index=False))
    if re_rows:
        print("\n=== 6. RE-INFERENCE ON FORMAT-NORMALIZED TEXT ===")
        print(pd.DataFrame(re_rows).round(4).to_string(index=False))
    print(f"\nAll outputs written to: {out_dir}")
    return 0


# ---- device-check: ModernBERT CPU versus stored GPU predictions on FRD
# -*- coding: utf-8 -*-



DEVICE_CHECK_KEYS = ("modernbert_base", "modernbert_sentiment_profile")


def _device_check_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def device_check_main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="score a random subset of this size (0 = all)")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args(argv)

    EXT._patch_torch_compile_for_windows()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = HERE / results_dir

    split = pd.read_csv(results_dir / "splits" / "frd_split.csv")
    id_col = "sample_id" if "sample_id" in split.columns else "source_index"
    test = split[split["split"] == "test"].copy()
    test["sid"] = test[id_col].astype(str)
    if args.limit and args.limit < len(test):
        test = test.sample(n=args.limit, random_state=args.seed)
    test = test.reset_index(drop=True)
    texts = test["text_"].astype(str).tolist()
    y = test["label_num"].to_numpy(dtype=int)
    print(f"FRD test reviews scored: {len(test)}")

    raw_features = None
    report = {"n_reviews": int(len(test)), "subset": bool(args.limit), "systems": {}}

    for key in DEVICE_CHECK_KEYS:
        bundle = EXT.resolve_model_bundle(results_dir, args.seed, key)
        is_hybrid = bundle.artifact_meta.get("type") == "hybrid"
        if is_hybrid and raw_features is None:
            print("Computing sentiment profiles for the FRD test reviews ...")
            raw_features = pd.DataFrame([EXT.compute_sentiment_profile(t) for t in texts])

        stored = pd.read_csv(bundle.result_dir / "test_predictions.csv")
        stored["sid"] = stored["sample_id"].astype(str)
        merged = test[["sid"]].merge(stored[["sid", "y_true", "y_prob_cg"]], on="sid", how="left")
        if merged["y_prob_cg"].isna().any():
            raise RuntimeError(f"{key}: stored predictions do not cover every test review")
        if (merged["y_true"].to_numpy(dtype=int) != y).any():
            raise RuntimeError(f"{key}: stored labels disagree with the split file")
        p_gpu = merged["y_prob_cg"].to_numpy(dtype=float)

        print(f"\n[{key}] CPU/FP32/eager inference ...")
        p_cpu, info = EXT.predict_bundle(
            bundle, texts, raw_features if is_hybrid else None, "cpu",
            args.max_length, args.batch, args.batch, modernbert_device="cpu",
        )
        p_cpu = np.asarray(p_cpu, dtype=float)
        tau = float(bundle.source_threshold)
        diff = np.abs(p_cpu - p_gpu)
        pred_gpu, pred_cpu = (p_gpu >= tau).astype(int), (p_cpu >= tau).astype(int)
        res = {
            "threshold": tau,
            "inference": {k: info.get(k) for k in ("device", "batch_size", "seconds")},
            "max_abs_prob_diff": float(diff.max()),
            "mean_abs_prob_diff": float(diff.mean()),
            "p99_abs_prob_diff": float(np.quantile(diff, 0.99)),
            "label_changes_at_threshold": int((pred_gpu != pred_cpu).sum()),
            "accuracy_stored_gpu": float((pred_gpu == y).mean()),
            "accuracy_cpu": float((pred_cpu == y).mean()),
            "roc_auc_stored_gpu": _device_check_auc(y, p_gpu),
            "roc_auc_cpu": _device_check_auc(y, p_cpu),
            "spearman_gpu_vs_cpu": float(pd.Series(p_gpu).corr(pd.Series(p_cpu), method="spearman")),
        }
        report["systems"][key] = res
        print(json.dumps(res, indent=2))

    out = results_dir / "paper_pack" / "device_check"
    out.mkdir(parents=True, exist_ok=True)
    (out / "modernbert_device_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out / 'modernbert_device_check.json'}")
    return 0

if __name__ == "__main__":
    _argv = sys.argv[1:]
    try:
        if _argv[:1] == ["diagnostics"]:
            raise SystemExit(diagnostics_main(_argv[1:]))
        if _argv[:1] == ["device-check"]:
            raise SystemExit(device_check_main(_argv[1:]))
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Existing FRD artifacts were not modified.", file=sys.stderr)
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(3)
