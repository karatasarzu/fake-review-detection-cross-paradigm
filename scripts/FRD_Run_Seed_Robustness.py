# -*- coding: utf-8 -*-
r"""FRD seed-robustness add-on for the backbone-matched sentiment ablation.

What it does
------------
Re-trains the three backbone pairs whose sentiment effect was significant in
the seed-42 study, each plain and with the 9-D sentiment profile, under
additional training seeds:

    BERT     : bert          vs bert_sentiment_profile
    RoBERTa  : roberta       vs roberta_sentiment_profile
    ELECTRA  : electra_base  vs electra_sentiment_profile

Everything except the training seed is identical to the seed-42 runs: the same
persisted split (results/splits/frd_split.csv), the same embedded runtime
(train_single_model / train_hybrid_model, PAYLOAD_SHA256 8ae96147...), the same
process supervisor, the same precision policy, the same sentiment features and
training-fitted scaler, the same checkpoint and threshold rules. The seed only
changes head initialization, data order and dropout. The script imports
FRD_Run_Transformer_Study.py and calls its own dispatch functions; it does not
modify any existing file or result.

Seeds
-----
The five-seed list 42, 123, 2025, 3407, 7777 was declared in
FRD_Revision_AllInOne.py before any multi-seed result existed.
    stage 1 : 123, 2025          (all three pairs; 12 fine-tuning runs)
    stage 2 : 3407, 7777         (only pairs flagged by the rule below)
Stage-2 rule (written to results/seed_robustness/protocol.json on first run,
before any new training): after stage 1, a pair receives seeds 3407 and 7777
if the sign of its test-accuracy difference (sentiment minus plain) is not the
same for seeds 42, 123 and 2025; a zero difference counts as inconsistent.

Commands (PowerShell, from the project folder)
----------------------------------------------
  .\.venv_frd_safe\Scripts\python.exe FRD_Run_Seed_Robustness.py plan
  .\.venv_frd_safe\Scripts\python.exe FRD_Run_Seed_Robustness.py train --dry-run
  .\.venv_frd_safe\Scripts\python.exe FRD_Run_Seed_Robustness.py all
  .\.venv_frd_safe\Scripts\python.exe FRD_Run_Seed_Robustness.py all --stage 2
  .\.venv_frd_safe\Scripts\python.exe FRD_Run_Seed_Robustness.py all --stage 3

Stage 3 (protocol amendment 1, recorded in protocol_amendment_1.json before it
runs): all seven backbone pairs with seeds 123, 2025, 3407 and 7777; verified
runs from stages 1 and 2 are reused. Each model keeps the numeric precision of
its own seed-42 run (DeBERTa + Sentiment FP32, gated DeBERTa-v3 FP16, plain
DeBERTa-v3 FP32, all others bf16); plain DeBERTa-v3 is retrained with the common
routine rather than the separate seed-42 routine.

  plan     writes the protocol file (once), then lists tasks, status and a time
           estimate from the seed-42 runs; trains nothing
  train    fine-tunes the missing runs (resumable; verified runs are skipped)
  aigen    zero-shot AiGen scoring of the new runs (inference only)
  analyze  seed-level tables for Table V / Table VI and the stage-2 decision
  all      train -> aigen -> analyze

Required beside this file: FRD_Run_Transformer_Study.py, the results folder of
the seed-42 study, AiGen test.csv (for aigen), and FRD_AiGen_External_Validation.py
(or FRD_AiGen_External_Validation_v1_0_3.py).

Outputs: results/models/frd/seed_<N>/<model>/            (training, same layout)
         results/external_aigen/seed_<N>/models/<model>/  (AiGen, same layout)
         results/seed_robustness/                        (protocol, tables, report)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import traceback

RUNNER_VERSION = "1.2.0"
PROTOCOL_VERSION = 1
REFERENCE_SEED = 42
DECLARED_SEEDS = (42, 123, 2025, 3407, 7777)
STAGE_SEEDS = {1: (123, 2025), 2: (3407, 7777), 3: (123, 2025, 3407, 7777)}

# Paper label -> (plain key, sentiment key). Order is the order of Table V.
PAIRS = {
    "BERT": ("bert", "bert_sentiment_profile"),
    "RoBERTa": ("roberta", "roberta_sentiment_profile"),
    "ELECTRA": ("electra_base", "electra_sentiment_profile"),
}
SENTIMENT_ENCODER = {
    "bert_sentiment_profile": "bert",
    "roberta_sentiment_profile": "roberta",
    "electra_sentiment_profile": "electra_base",
    "deberta_sentiment_profile": "deberta",
    "deberta_v3_sentiment_gated": "deberta_v3_base",
    "modernbert_sentiment_profile": "modernbert_base",
    "distilbert_sentiment_profile": "distilbert",
}
GATED_KEYS = {"deberta_v3_sentiment_gated"}
# Stage 3 (protocol amendment 1): all seven backbone pairs, all five seeds.
ALL_PAIRS = {
    "BERT": ("bert", "bert_sentiment_profile"),
    "RoBERTa": ("roberta", "roberta_sentiment_profile"),
    "DeBERTa": ("deberta", "deberta_sentiment_profile"),
    "DeBERTa-v3": ("deberta_v3_base", "deberta_v3_sentiment_gated"),
    "ModernBERT": ("modernbert_base", "modernbert_sentiment_profile"),
    "ELECTRA": ("electra_base", "electra_sentiment_profile"),
    "DistilBERT": ("distilbert", "distilbert_sentiment_profile"),
}
AMENDMENT_1 = {
    "amendment": 1,
    "scope": "stage 3: all seven backbone pairs with seeds 123, 2025, 3407 and 7777 (runs already present are reused)",
    "reason": ("Decided after stages 1 and 2 so that every pair in Table V has five seeds. The four added pairs "
               "had not been retrained before this amendment; ELECTRA is completed to five seeds."),
    "precision_policy": "each model uses the numeric precision recorded in its own seed-42 run",
    "deberta_v3_note": ("plain DeBERTa-v3 is retrained with the common training routine in FP32; its seed-42 run "
                        "came from a separate routine"),
}
STAGE2_RULE = (
    "After stage 1, a pair receives seeds 3407 and 7777 if the sign of its test-accuracy "
    "difference (sentiment minus plain) is not the same for seeds 42, 123 and 2025; a zero "
    "difference counts as inconsistent. No other pair is extended."
)
# Settings that must match the seed-42 run of the same model.
CONSISTENCY_FIELDS = ("profile", "micro_batch", "gradient_accumulation", "effective_batch",
                      "precision", "learning_rate", "weight_decay", "warmup_ratio",
                      "max_length", "max_epochs", "adaptation")
AIGEN_MD5 = "c41a361eafa9800e3e8be3362a0101f4"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        return default


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(str(tmp), str(path))


def all_model_keys(pairs):
    return [k for label in pairs for k in ALL_PAIRS[label]]


def task_key(seed, key):
    kind = "hybrid" if key in SENTIMENT_ENCODER else "single"
    return f"frd:seed{seed}:{kind}:{key}"


def safe_python_path(project):
    if sys.platform.lower().startswith("win"):
        return Path(project) / ".venv_frd_safe" / "Scripts" / "python.exe"
    return Path(project) / ".venv_frd_safe" / "bin" / "python"


def ensure_safe_interpreter(project, argv):
    """Relaunch THIS script with the project-local .venv_frd_safe interpreter."""
    if os.environ.get("FRD_SEEDROB_SKIP_ENV_CHECK") == "1":  # tests only
        return
    expected = safe_python_path(project)
    try:
        same = os.path.normcase(str(Path(sys.executable).resolve())) == os.path.normcase(str(expected.resolve()))
    except Exception:
        same = False
    if same:
        return
    if not expected.is_file():
        raise RuntimeError("Project-local environment not found: " + str(expected) +
                           "\nUse the same .venv_frd_safe that trained the seed-42 models.")
    if os.environ.get("FRD_SEEDROB_REEXEC") == "1":
        raise RuntimeError(f"Relaunch did not reach the safe interpreter.\nExpected: {expected}\nActual: {sys.executable}")
    print("[SAFE ENV] Relaunching with:", expected, flush=True)
    env = os.environ.copy()
    env["FRD_SEEDROB_REEXEC"] = "1"
    completed = subprocess.run([str(expected), str(Path(__file__).resolve())] + list(argv), env=env)
    raise SystemExit(int(completed.returncode))


def load_study_module(project):
    """Import FRD_Run_Transformer_Study.py under its own module name.

    Registering the module name matters: the supervisor's _load_supervisor()
    looks itself up in sys.modules, and worker processes import it by name.
    """
    path = Path(project) / "FRD_Run_Transformer_Study.py"
    if not path.is_file():
        raise FileNotFoundError("Place this script beside FRD_Run_Transformer_Study.py: " + str(path))
    name = "FRD_Run_Transformer_Study"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for attr in ("Dispatch", "original_runtime", "_dispatch_single", "_dispatch_hybrid",
                 "_se_read_split", "inspect_training_bundle", "ProjectLock", "worker_env"):
        if not hasattr(module, attr):
            raise RuntimeError(f"FRD_Run_Transformer_Study.py lacks {attr}; unexpected version.")
    return module


def load_aigen_module(project):
    for fname in ("FRD_AiGen_External_Validation.py", "FRD_AiGen_External_Validation_v1_0_3.py"):
        path = Path(project) / fname
        if path.is_file():
            name = "FRD_AiGen_External_Validation"
            if name in sys.modules:
                return sys.modules[name]
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("FRD_AiGen_External_Validation.py not found beside this script.")


# -----------------------------------------------------------------------------
# Result bundles (read-only, no runtime import needed)
# -----------------------------------------------------------------------------

def resolve_bundle(project, seed, key):
    """Return {'metrics', 'result_dir', 'artifact', 'test_path', 'val_path'} or None."""
    root = Path(project) / "results" / "models" / "frd" / f"seed_{seed}" / key
    sel = read_json(root / "selected_result.json")
    if not isinstance(sel, dict):
        return None
    candidates = []
    raw = sel.get("result_dir")
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else Path(project) / p)
    if sel.get("profile"):
        candidates.append(root / str(sel["profile"]))
    candidates.append(root / "full_primary")
    candidates += sorted(p.parent for p in root.glob("*/metrics.json"))
    result_dir = next((c for c in candidates
                       if (c / "metrics.json").is_file() and (c / "test_predictions.csv").is_file()
                       and (c / "validation_predictions.csv").is_file()), None)
    if result_dir is None:
        return None
    metrics = read_json(result_dir / "metrics.json", {}) or {}
    for k in ("selected_threshold", "validation_default", "test_default", "test_calibrated", "resource"):
        if k not in metrics and k in sel:
            metrics[k] = sel[k]
    artifact = result_dir / "artifact"
    return {"metrics": metrics, "result_dir": result_dir, "artifact": artifact,
            "test_path": result_dir / "test_predictions.csv",
            "val_path": result_dir / "validation_predictions.csv", "selected": sel}


def seed42_train_minutes(project, key):
    b = resolve_bundle(project, REFERENCE_SEED, key)
    if not b:
        return None
    sec = (b["metrics"].get("resource") or {}).get("train_seconds")
    return float(sec) / 60.0 if isinstance(sec, (int, float)) else None


# -----------------------------------------------------------------------------
# Protocol (pre-registration)
# -----------------------------------------------------------------------------

def protocol_path(project):
    return Path(project) / "results" / "seed_robustness" / "protocol.json"


def build_protocol(project):
    split = Path(project) / "results" / "splits" / "frd_split.csv"
    if not split.is_file():
        raise FileNotFoundError("Persisted FRD split not found: " + str(split))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "runner_version": RUNNER_VERSION,
        "reference_seed": REFERENCE_SEED,
        "declared_seed_list": list(DECLARED_SEEDS),
        "declared_seed_list_source": "FRD_Revision_AllInOne.py docstring (FRD_TRAIN_SEEDS example)",
        "stage_1_seeds": list(STAGE_SEEDS[1]),
        "stage_2_seeds": list(STAGE_SEEDS[2]),
        "pairs": {k: list(v) for k, v in PAIRS.items()},
        "pair_choice": "the three pairs with a significant sentiment effect at seed 42 (Holm-corrected McNemar)",
        "stage_2_rule": STAGE2_RULE,
        "primary_quantity": "test accuracy difference, sentiment minus plain, at each model's validation-selected threshold",
        "unchanged": "split, runtime, supervisor, precision policy, features, scaler, hyperparameters, checkpoint and threshold rules",
        "frd_split_sha256": sha256_file(split),
    }


FIXED_PROTOCOL_FIELDS = ("protocol_version", "reference_seed", "declared_seed_list", "stage_1_seeds",
                         "stage_2_seeds", "pairs", "stage_2_rule", "primary_quantity", "frd_split_sha256")


def ensure_protocol(project, write=True):
    """Create the protocol file once; afterwards refuse silent changes."""
    path = protocol_path(project)
    new = build_protocol(project)
    old = read_json(path)
    if isinstance(old, dict):
        diffs = [f for f in FIXED_PROTOCOL_FIELDS if old.get(f) != new.get(f)]
        if diffs:
            raise RuntimeError("The pre-registered protocol differs from the current setup in: " +
                               ", ".join(diffs) + f"\nProtocol file: {path}\nNo run was started.")
        return old
    if write:
        new["created_utc"] = now()
        write_json(path, new)
        print("[PROTOCOL] Written before any new training:", path, flush=True)
    return new


def stage_plan(project, stage):
    """Seeds and pairs for a stage. Stage 2 depends on the stage-1 decision file."""
    if stage == 1:
        return list(STAGE_SEEDS[1]), list(PAIRS)
    if stage == 3:
        return list(STAGE_SEEDS[3]), list(ALL_PAIRS)
    decision = read_json(Path(project) / "results" / "seed_robustness" / "stage2_decision.json")
    if not isinstance(decision, dict) or not decision.get("stage_1_complete"):
        raise RuntimeError("Stage 2 needs the stage-1 decision. Finish stage 1 and run 'analyze' first.")
    return list(STAGE_SEEDS[2]), list(decision.get("extend_pairs", []))


# -----------------------------------------------------------------------------
# plan
# -----------------------------------------------------------------------------

def ensure_amendment(project):
    """Record amendment 1 once, before any stage-3 training."""
    path = Path(project) / "results" / "seed_robustness" / "protocol_amendment_1.json"
    old = read_json(path)
    if isinstance(old, dict):
        return old
    rec = dict(AMENDMENT_1)
    rec["recorded_utc"] = now()
    done = {}
    for label, (plain, sent) in ALL_PAIRS.items():
        done[label] = [s for s in STAGE_SEEDS[3]
                       if resolve_bundle(project, s, plain) and resolve_bundle(project, s, sent)]
    rec["pairs_with_repeated_runs_at_recording"] = done
    write_json(path, rec)
    print("[AMENDMENT] Recorded before stage-3 training:", path, flush=True)
    return rec


def parse_forced_precision(items):
    out = {}
    for item in items or []:
        key, _, prec = str(item).partition("=")
        if prec not in ("bf16", "fp16", "fp32") or key not in {k for v in ALL_PAIRS.values() for k in v}:
            raise SystemExit(f"--force-precision expects MODEL=bf16|fp16|fp32 with a known model key, got {item!r}")
        out[key] = prec
    return out


def ensure_amendment_2(project, forced):
    """Record a precision deviation from the seed-42 run once, before it is used."""
    if not forced:
        return None
    path = Path(project) / "results" / "seed_robustness" / "protocol_amendment_2.json"
    old = read_json(path)
    if isinstance(old, dict):
        if old.get("forced_precision") != forced:
            raise RuntimeError(f"protocol_amendment_2.json already records {old.get('forced_precision')}; refusing {forced}.")
        return old
    rec = {"amendment": 2, "forced_precision": forced,
           "reason": ("bf16 training of plain DeBERTa failed at the first step in the pinned environment "
                      "(masked_fill: value cannot be converted to BFloat16 without overflow); the repeated plain "
                      "DeBERTa runs therefore use FP32, the precision of the DeBERTa sentiment variant"),
           "recorded_utc": now()}
    write_json(path, rec)
    print("[AMENDMENT 2] Recorded:", path, flush=True)
    return rec


def reference_precision(project, key):
    b = resolve_bundle(project, REFERENCE_SEED, key)
    prec = ((b or {}).get("metrics", {}).get("resource") or {}).get("precision")
    return prec if prec in ("bf16", "fp16", "fp32") else None


def cmd_plan(project, stage):
    S = load_study_module(project)
    ensure_protocol(project, write=True)
    if stage == 3:
        ensure_amendment(project)
    try:
        seeds, pairs = stage_plan(project, stage)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    print(f"\nStage {stage}: seeds {seeds}; pairs {pairs}\n")
    total = 0.0
    missing_ref = []
    for label in pairs:
        for key in ALL_PAIRS[label]:
            if not resolve_bundle(project, REFERENCE_SEED, key):
                missing_ref.append(key)
    if missing_ref:
        print("[ERROR] Seed-42 reference runs missing:", missing_ref)
        return 1
    for seed in seeds:
        for label in pairs:
            for key in ALL_PAIRS[label]:
                tk = task_key(seed, key)
                status = S.inspect_training_bundle(project, tk)
                done = "DONE" if status.get("complete") else ("PARTIAL/SAVED" if status.get("saved_model") else "todo")
                minutes = seed42_train_minutes(project, key)
                if done != "DONE" and minutes:
                    total += minutes
                print(f"  {tk:45s} {done:14s} seed-42 time: {minutes:.0f} min" if minutes
                      else f"  {tk:45s} {done:14s} seed-42 time: n/a")
    print(f"\nEstimated remaining training time (from seed-42 train_seconds): {total/60:.1f} h")
    print("AiGen scoring adds a few minutes per model; analysis is instantaneous.")
    return 0


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------

def _check_scaler_matches_reference(project, scaler, feature_names):
    """The sentiment scaler must be the one stored in the seed-42 hybrid artifacts."""
    import numpy as np
    for key in SENTIMENT_ENCODER:
        b = resolve_bundle(project, REFERENCE_SEED, key)
        if not b:
            continue
        meta = read_json(Path(b["artifact"]) / "artifact_meta.json", {}) or {}
        if "scaler_mean" not in meta:
            continue
        if list(meta.get("feature_names", [])) != list(feature_names):
            raise RuntimeError(f"Feature list differs from the seed-42 artifact of {key}.")
        ok = (np.allclose(meta["scaler_mean"], scaler.mean_, rtol=0, atol=1e-9) and
              np.allclose(meta["scaler_scale"], scaler.scale_, rtol=0, atol=1e-9))
        if not ok:
            raise RuntimeError(f"Sentiment scaler differs from the seed-42 artifact of {key}; refusing to train.")
        return key
    raise RuntimeError("No seed-42 sentiment artifact with a stored scaler was found for comparison.")


def cmd_train(project, stage, native_retries, max_task_hours, dry_run, forced=None):
    forced = forced or {}
    S = load_study_module(project)
    protocol = ensure_protocol(project, write=True)
    if stage == 3:
        ensure_amendment(project)
    seeds, pairs = stage_plan(project, stage)
    if not pairs:
        print("[STAGE 2] The stage-1 rule flagged no pair; nothing to train.")
        return 0
    for label in pairs:
        for key in ALL_PAIRS[label]:
            if not resolve_bundle(project, REFERENCE_SEED, key):
                raise RuntimeError(f"Seed-42 reference run missing for {key}; refusing to start.")

    with S.ProjectLock(Path(project) / ".frd_supervisor" / "runner.lock"):
        runtime, digest = S.original_runtime(project)
        session = Path(project) / ".frd_supervisor" / "sessions" / ("seedrob_" + S.stamp())
        session.mkdir(parents=True, exist_ok=True)
        results_dir = Path(project) / "results"
        S.backup_database(results_dir / "state.db", session / "state_before.sqlite3")
        os.environ.update(S.worker_env(project))
        sys.path.insert(0, str(runtime))

        from config import CONFIG
        from frdexp.features import FEATURE_NAMES, build_feature_cache, fit_feature_scaler
        from frdexp.model_registry import MODEL_SPECS, HYBRID_SPECS
        from frdexp.state import StateDB

        frame, text_col, split_path = S._se_read_split(project, "frd")
        if sha256_file(split_path) != protocol["frd_split_sha256"]:
            raise RuntimeError("frd_split.csv changed since the protocol was written.")
        cache = results_dir / "feature_cache" / "frd_sentiment.csv"
        if not cache.is_file():
            raise FileNotFoundError("Sentiment feature cache of the seed-42 study not found: " + str(cache))
        features = build_feature_cache(frame, text_col, cache)
        if len(features) != len(frame):
            raise RuntimeError("Feature cache rows do not match the split rows.")
        scaler = fit_feature_scaler(features.loc[frame.split == "train"])  # not written to disk
        checked = _check_scaler_matches_reference(project, scaler, FEATURE_NAMES)

        hybrid_specs = {}
        for key, enc_key in SENTIMENT_ENCODER.items():
            if key in HYBRID_SPECS:
                hybrid_specs[key] = dict(HYBRID_SPECS[key])
            else:  # identical to FRD_Run_Transformer_Study.sentiment_main
                hybrid_specs[key] = {"paper_name": MODEL_SPECS[enc_key].paper_name + " + Sentiment Profile",
                                     "encoder": enc_key, "features": "profile", "gated": key in GATED_KEYS}
            hs = hybrid_specs[key]
            if hs["encoder"] != enc_key or hs["features"] != "profile" or bool(hs["gated"]) != (key in GATED_KEYS):
                raise RuntimeError(f"Unexpected hybrid specification for {key}: {hs}")

        tasks = [(seed, key) for seed in seeds for label in pairs for key in ALL_PAIRS[label]]
        precision = {}
        for _, key in tasks:
            if key not in precision:
                precision[key] = forced.get(key) or reference_precision(project, key)
                if precision[key] is None:
                    raise RuntimeError(f"Seed-42 precision of {key} not recorded; refusing to guess.")
        print(f"FRD seed-robustness runner {RUNNER_VERSION} | stage {stage}", flush=True)
        print("PYTHON:", sys.executable, flush=True)
        print("RUNTIME:", digest, flush=True)
        print("SPLIT:", split_path, "| rows", len(frame), flush=True)
        print("SCALER: identical to seed-42 artifact of", checked, flush=True)
        print("CONFIG: max_length", CONFIG.max_length, "| max_epochs", CONFIG.max_epochs,
              "| effective batch", CONFIG.effective_batch_size, flush=True)
        print("TASKS:", flush=True)
        for seed, key in tasks:
            status = "done" if S.inspect_training_bundle(project, task_key(seed, key)).get("complete") else "todo"
            print("  ", task_key(seed, key), f"[{precision[key]}]", status, flush=True)
        if dry_run:
            print("\n[DRY RUN] All checks passed; no training started.", flush=True)
            return 0
        ensure_amendment_2(project, forced)

        dispatch = S.Dispatch(project, runtime, session, native_retries=native_retries,
                              timeout=max_task_hours * 3600)
        state = StateDB(results_dir / "state.db")
        try:
            for seed, key in tasks:
                # Same callables as FRD_Run_Transformer_Study, but the precision is the one
                # recorded in this model's own seed-42 run (bf16 unless the paper states otherwise).
                prec = precision[key]
                tk = task_key(seed, key)
                root = results_dir / "models" / "frd" / f"seed_{seed}" / key
                print(f"\n=== {tk} [{prec}] ===", flush=True)
                if key in SENTIMENT_ENCODER:
                    enc = MODEL_SPECS[SENTIMENT_ENCODER[key]]
                    hs = hybrid_specs[key]
                    call = S._hybrid_training_callable(key, hs, enc, frame, text_col, features,
                                                       FEATURE_NAMES, scaler, seed, CONFIG, root,
                                                       precision_override=prec)
                    payload = {"protocol": "frd", "model": key, "seed": seed, "encoder": enc.hf_id,
                               "hf_id": enc.hf_id, "scope": "seed_robustness", "sentiment_features": "profile_9d",
                               "fusion": "gated" if hs.get("gated") else "late_concat", "precision_policy": prec}
                else:
                    spec = MODEL_SPECS[key]
                    call = S._single_training_callable(spec, frame, text_col, seed, CONFIG, root,
                                                       precision_override=prec)
                    payload = {"protocol": "frd", "model": key, "seed": seed, "hf_id": spec.hf_id,
                               "scope": "seed_robustness", "precision_policy": prec}
                dispatch(state, tk, call, payload=payload)
        finally:
            state.close()
        failed = sorted(set(dispatch.failures))
        summary = {"runner_version": RUNNER_VERSION, "stage": stage, "seeds": seeds, "pairs": pairs,
                   "failed_tasks": failed, "session": str(session), "time": now()}
        write_json(results_dir / "seed_robustness" / f"train_stage{stage}_summary.json", summary)
        print("\n=== TRAINING SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
        return 2 if failed else 0


# -----------------------------------------------------------------------------
# aigen (inference only)
# -----------------------------------------------------------------------------

def cmd_aigen(project, stage, dataset, device):
    import pandas as pd
    EXT = load_aigen_module(project)
    EXT._patch_torch_compile_for_windows()
    seeds, pairs = stage_plan(project, stage)
    if not pairs:
        print("[STAGE 2] No pair to score.")
        return 0
    dataset = Path(dataset)
    if not dataset.is_absolute():
        dataset = Path(project) / dataset
    results_dir = Path(project) / "results"
    raw_df, frame = EXT.load_aigen_test(dataset)
    md5 = EXT._md5(dataset)
    if md5 != AIGEN_MD5:
        raise RuntimeError(f"AiGen test.csv MD5 {md5} differs from the official release {AIGEN_MD5}.")
    sha = EXT._sha256(dataset)
    shared = results_dir / "external_aigen" / f"seed_{REFERENCE_SEED}" / "aigen_sentiment_features.csv"
    cache = shared if (shared.is_file() and shared.with_suffix(".meta.json").is_file()) \
        else results_dir / "seed_robustness" / "aigen_sentiment_features.csv"
    raw_features = EXT.build_sentiment_features(frame, cache, sha)
    texts = frame["text"].astype(str).tolist()
    y = frame["label_num"].to_numpy(dtype=int)
    dev = EXT._device_from_arg(device)
    failures = {}
    for seed in seeds:
        for label in pairs:
            for key in ALL_PAIRS[label]:
                target = results_dir / "external_aigen" / f"seed_{seed}" / "models" / key
                pred_path, metrics_path = target / "predictions.csv", target / "metrics.json"
                display = EXT.MODEL_DISPLAY_NAMES.get(key, key)
                if pred_path.is_file() and metrics_path.is_file():
                    old = pd.read_csv(pred_path)
                    if len(old) == len(frame) and old["sample_id"].astype(str).tolist() == frame["source_index"].astype(str).tolist():
                        print(f"[CACHE] seed {seed} {display}")
                        continue
                try:
                    bundle = EXT.resolve_model_bundle(results_dir, seed, key)
                    print(f"[RUN]   seed {seed} {display} (threshold {bundle.source_threshold:.4f})", flush=True)
                    prob, perf = EXT.predict_bundle(bundle, texts, raw_features, dev, 256, 16, 8,
                                                    modernbert_device="cpu")
                    primary, default = EXT.system_metric_rows(f"model::{key}", display, "model",
                                                              bundle.source_threshold, y, prob, inference=perf)
                    EXT._json_dump(metrics_path, {
                        "protocol": "zero_shot_frd_to_aigen", "dataset": "AiGen-FoodReview official test split",
                        "dataset_sha256": sha, "model_key": key, "system": display, "seed": seed,
                        "artifact": str(bundle.artifact.resolve()), "source_threshold": bundle.source_threshold,
                        "threshold_source": "FRD_validation_only",
                        "primary_frd_validation_threshold": primary, "secondary_default_0_5": default,
                        "inference": perf, "runner": f"FRD_Run_Seed_Robustness {RUNNER_VERSION}"})
                    EXT.save_prediction_frame(pred_path, frame, prob, bundle.source_threshold,
                                              f"model::{key}", display, seed)
                    print(f"        AUC={primary['roc_auc']:.4f} Acc={primary['accuracy']:.4f}", flush=True)
                except Exception as exc:
                    failures[f"seed{seed}:{key}"] = repr(exc)
                    traceback.print_exc()
                    EXT._cleanup_cuda()
    if failures:
        print("[AIGEN FAILURES]", json.dumps(failures, indent=2))
        return 2
    return 0


# -----------------------------------------------------------------------------
# analyze (no training, no selection)
# -----------------------------------------------------------------------------

def _load_test_frame(bundle):
    import pandas as pd
    d = pd.read_csv(bundle["test_path"])
    if "sample_id" not in d.columns and "source_index" in d.columns:
        d = d.rename(columns={"source_index": "sample_id"})
    d["sample_id"] = d["sample_id"].astype(str)
    return d[["sample_id", "y_true", "y_prob_cg"]].sort_values("sample_id").reset_index(drop=True)


def _metrics(y, p, threshold):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    pred = (np.asarray(p) >= float(threshold)).astype(int)
    y = np.asarray(y, dtype=int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {"accuracy": (tp + tn) / len(y), "f1": 2 * prec * rec / max(1e-15, prec + rec),
            "roc_auc": float(roc_auc_score(y, p)), "recall": rec, "flagged": tp + fp,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "pred": pred}


def _mcnemar(y, pred_a, pred_b):
    """Identical rule to the paper: continuity-corrected, exact binomial below 25 discordant."""
    import numpy as np
    from scipy.stats import binomtest, chi2
    ca, cb = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = b + c
    if n == 0:
        return 1.0, b, c
    if n < 25:
        return float(binomtest(min(b, c), n, 0.5).pvalue), b, c
    return float(chi2.sf((abs(b - c) - 1) ** 2 / n, 1)), b, c


def _summary_stats(values):
    import numpy as np
    from scipy.stats import t
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean()) if n else float("nan")
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        half = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n))
        lo, hi = mean - half, mean + half
    else:
        lo = hi = float("nan")
    return {"n": n, "mean": mean, "sd": sd, "ci_low": lo, "ci_high": hi,
            "min": float(v.min()) if n else float("nan"), "max": float(v.max()) if n else float("nan")}


def cmd_analyze(project):
    import numpy as np
    import pandas as pd
    results_dir = Path(project) / "results"
    out = results_dir / "seed_robustness"
    out.mkdir(parents=True, exist_ok=True)
    protocol = ensure_protocol(project, write=False)
    all_seeds = list(DECLARED_SEEDS)

    per_seed, per_model, warnings = [], [], []
    ref_resource = {}
    for label, (plain, sent) in ALL_PAIRS.items():
        for key in (plain, sent):
            b = resolve_bundle(project, REFERENCE_SEED, key)
            ref_resource[key] = (b["metrics"].get("resource") or {}) if b else {}
        for seed in all_seeds:
            ba, bb = resolve_bundle(project, seed, plain), resolve_bundle(project, seed, sent)
            for key, b in ((plain, ba), (sent, bb)):
                if not b:
                    continue
                res = b["metrics"].get("resource") or {}
                if seed != REFERENCE_SEED:
                    for f in CONSISTENCY_FIELDS:
                        if f in ref_resource[key] and res.get(f) != ref_resource[key].get(f):
                            warnings.append(f"seed {seed} {key}: {f}={res.get(f)!r} vs seed 42 {ref_resource[key].get(f)!r}")
                    if int(b["metrics"].get("seed", seed)) != seed:
                        warnings.append(f"seed {seed} {key}: metrics.json reports seed {b['metrics'].get('seed')}")
                d = _load_test_frame(b)
                m = _metrics(d.y_true.to_numpy(int), d.y_prob_cg.to_numpy(float), b["metrics"]["selected_threshold"])
                per_model.append({"seed": seed, "backbone": label, "model": key,
                                  "variant": "sentiment" if key == sent else "plain",
                                  "validation_accuracy_0.5": (b["metrics"].get("validation_default") or {}).get("accuracy"),
                                  "threshold": b["metrics"]["selected_threshold"],
                                  "test_accuracy": m["accuracy"], "test_f1": m["f1"], "test_roc_auc": m["roc_auc"],
                                  "errors": m["fp"] + m["fn"], "precision_mode": res.get("precision"),
                                  "train_minutes": (res.get("train_seconds") or 0) / 60.0})
            if not (ba and bb):
                continue
            da, db = _load_test_frame(ba), _load_test_frame(bb)
            if not (da.sample_id.equals(db.sample_id) and da.y_true.equals(db.y_true)):
                raise RuntimeError(f"Test predictions not aligned for {label} seed {seed}.")
            y = da.y_true.to_numpy(int)
            ma = _metrics(y, da.y_prob_cg.to_numpy(float), ba["metrics"]["selected_threshold"])
            mb = _metrics(y, db.y_prob_cg.to_numpy(float), bb["metrics"]["selected_threshold"])
            p, b_only, c_only = _mcnemar(y, ma["pred"], mb["pred"])
            per_seed.append({"backbone": label, "seed": seed,
                             "acc_plain": ma["accuracy"], "acc_sentiment": mb["accuracy"],
                             "delta_acc_pp": 100 * (mb["accuracy"] - ma["accuracy"]),
                             "delta_f1_pp": 100 * (mb["f1"] - ma["f1"]),
                             "delta_auc_x100": 100 * (mb["roc_auc"] - ma["roc_auc"]),
                             "plain_only_correct": b_only, "sentiment_only_correct": c_only,
                             "mcnemar_p_unadjusted": p})

    seed_df = pd.DataFrame(per_seed)
    model_df = pd.DataFrame(per_model)
    seed_df.to_csv(out / "per_seed_pair_differences.csv", index=False)
    model_df.to_csv(out / "per_seed_model_metrics.csv", index=False)
    if seed_df.empty:
        print("No complete pair found.")
        return 1

    # Pair summary across available seeds
    summary_rows = []
    for label in ALL_PAIRS:
        g = seed_df[seed_df.backbone == label].sort_values("seed")
        if g.empty:
            continue
        row = {"backbone": label, "seeds": ",".join(str(s) for s in g.seed)}
        for col in ("delta_acc_pp", "delta_f1_pp", "delta_auc_x100"):
            st = _summary_stats(g[col])
            row.update({f"{col}_{k}": v for k, v in st.items() if k != "n"})
        row["n_seeds"] = len(g)
        ref = g[g.seed == REFERENCE_SEED].delta_acc_pp
        ref_sign = float(np.sign(ref.iloc[0])) if len(ref) else float("nan")
        signs = np.sign(g.delta_acc_pp.to_numpy())
        row["seed42_delta_acc_pp"] = float(ref.iloc[0]) if len(ref) else float("nan")
        row["same_sign_as_seed42"] = int((signs == ref_sign).sum()) if len(ref) else 0
        row["sign_consistent"] = bool(len(ref) and ref_sign != 0 and (signs == ref_sign).all())
        for variant in ("plain", "sentiment"):
            gm = model_df[(model_df.backbone == label) & (model_df.variant == variant)]
            st = _summary_stats(gm.test_accuracy)
            row[f"acc_{variant}_mean"] = st["mean"]; row[f"acc_{variant}_sd"] = st["sd"]
        summary_rows.append(row)
    pair_df = pd.DataFrame(summary_rows)
    pair_df.to_csv(out / "pair_summary.csv", index=False)

    # Stage-2 decision (written only once stage 1 is complete for every pair)
    stage1 = (REFERENCE_SEED,) + tuple(STAGE_SEEDS[1])
    complete = all(set(stage1) <= set(seed_df[seed_df.backbone == lb].seed) for lb in PAIRS)
    decision_path = out / "stage2_decision.json"
    if complete:
        extend = []
        for lb in PAIRS:
            g = seed_df[(seed_df.backbone == lb) & (seed_df.seed.isin(stage1))]
            s = np.sign(g.delta_acc_pp.to_numpy())
            if (s == 0).any() or len(set(s.tolist())) > 1:
                extend.append(lb)
        decision = {"stage_1_complete": True, "rule": STAGE2_RULE, "extend_pairs": extend,
                    "stage_2_seeds": list(STAGE_SEEDS[2]), "decided_utc": now(),
                    "stage_1_deltas_pp": {lb: seed_df[(seed_df.backbone == lb) & (seed_df.seed.isin(stage1))]
                                          .set_index("seed").delta_acc_pp.round(4).to_dict() for lb in PAIRS}}
        old = read_json(decision_path)
        if isinstance(old, dict) and old.get("extend_pairs") != extend:
            warnings.append("stage2_decision.json already existed with a different decision; kept the original.")
            decision = old
        elif not isinstance(old, dict):
            write_json(decision_path, decision)
    else:
        decision = {"stage_1_complete": False}

    # AiGen seed summary (reads stored predictions; labels used only for metrics)
    aig_rows = []
    for label, (plain, sent) in ALL_PAIRS.items():
        for key in (plain, sent):
            for seed in all_seeds:
                pp = results_dir / "external_aigen" / f"seed_{seed}" / "models" / key / "predictions.csv"
                if not pp.is_file():
                    continue
                d = pd.read_csv(pp)
                m = _metrics(d.y_true.to_numpy(int), d.y_prob_machine_generated.to_numpy(float), float(d.threshold.iloc[0]))
                aig_rows.append({"backbone": label, "model": key, "seed": seed, "roc_auc": m["roc_auc"],
                                 "accuracy": m["accuracy"], "recall": m["recall"], "flagged": m["flagged"],
                                 "threshold": float(d.threshold.iloc[0])})
    aig_df = pd.DataFrame(aig_rows)
    aig_df.to_csv(out / "aigen_per_seed.csv", index=False)
    aig_sum = []
    if not aig_df.empty:
        for key, g in aig_df.groupby("model", sort=False):
            st = _summary_stats(g.roc_auc)
            aig_sum.append({"model": key, "seeds": ",".join(str(s) for s in sorted(g.seed)),
                            "auc_mean": st["mean"], "auc_sd": st["sd"], "auc_min": st["min"], "auc_max": st["max"],
                            "seeds_auc_above_0.5": int((g.roc_auc > 0.5).sum()),
                            "max_flagged": int(g.flagged.max()), "max_recall": float(g.recall.max())})
    aig_sum_df = pd.DataFrame(aig_sum)
    aig_sum_df.to_csv(out / "aigen_seed_summary.csv", index=False)

    # LaTeX rows and report
    tex = ["% Table V, seed columns: backbone & mean dAcc (pp) +- SD & min..max & same sign as seed 42"]
    for _, r in pair_df.iterrows():
        tex.append(f"{r.backbone} & ${r.delta_acc_pp_mean:+.2f} \\pm {r.delta_acc_pp_sd:.2f}$ & "
                   f"[${r.delta_acc_pp_min:+.2f}$, ${r.delta_acc_pp_max:+.2f}$] & "
                   f"{int(r.same_sign_as_seed42)}/{int(r.n_seeds)} \\\\")
    (out / "table_v_seed_rows.tex").write_text("\n".join(tex), encoding="utf-8")

    lines = ["# Seed robustness of the sentiment ablation", "",
             f"Protocol written {protocol.get('created_utc', 'n/a')}; stage-2 rule: {STAGE2_RULE}", "",
             "## Test-accuracy difference, sentiment minus plain (pp)", "",
             "| Backbone | Seeds | Seed 42 | Mean ± SD | 95% t-CI | Min–max | Same sign as seed 42 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in pair_df.iterrows():
        lines.append(f"| {r.backbone} | {r.seeds} | {r.seed42_delta_acc_pp:+.2f} | "
                     f"{r.delta_acc_pp_mean:+.2f} ± {r.delta_acc_pp_sd:.2f} | "
                     f"[{r.delta_acc_pp_ci_low:+.2f}, {r.delta_acc_pp_ci_high:+.2f}] | "
                     f"{r.delta_acc_pp_min:+.2f} … {r.delta_acc_pp_max:+.2f} | "
                     f"{int(r.same_sign_as_seed42)}/{int(r.n_seeds)} |")
    lines += ["", "## Per seed", "", "| Backbone | Seed | Plain | Sentiment | ΔAcc (pp) | ΔF1 (pp) | ΔAUC ×100 | McNemar p (unadj.) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in seed_df.sort_values(["backbone", "seed"]).iterrows():
        lines.append(f"| {r.backbone} | {r.seed} | {r.acc_plain:.4f} | {r.acc_sentiment:.4f} | {r.delta_acc_pp:+.2f} | "
                     f"{r.delta_f1_pp:+.2f} | {r.delta_auc_x100:+.2f} | {r.mcnemar_p_unadjusted:.3g} |")
    if not aig_sum_df.empty:
        lines += ["", "## AiGen ROC-AUC across seeds", "", "| Model | Seeds | Mean ± SD | Min–max | Seeds with AUC > 0.5 |",
                  "|---|---|---:|---:|---:|"]
        for _, r in aig_sum_df.iterrows():
            sd = "n/a" if not np.isfinite(r.auc_sd) else f"{r.auc_sd:.4f}"
            lines.append(f"| {r.model} | {r.seeds} | {r.auc_mean:.4f} ± {sd} | {r.auc_min:.4f}–{r.auc_max:.4f} | "
                         f"{int(r['seeds_auc_above_0.5'])}/{len(r.seeds.split(','))} |")
    lines += ["", "## Stage-2 decision", "", "```", json.dumps(decision, indent=2, default=str), "```"]
    if warnings:
        lines += ["", "## Consistency warnings", ""] + [f"- {w}" for w in warnings]
    else:
        lines += ["", "All new runs used the same profile, precision and hyperparameters as seed 42."]
    (out / "seed_robustness_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nOutputs:", out)
    if decision.get("stage_1_complete") and decision.get("extend_pairs"):
        print("\n[NEXT] The stage-2 rule flagged:", ", ".join(decision["extend_pairs"]))
        print("       Run: .\\.venv_frd_safe\\Scripts\\python.exe FRD_Run_Seed_Robustness.py all --stage 2")
    elif decision.get("stage_1_complete"):
        print("\n[DONE] No pair meets the stage-2 rule; the seed study is complete.")
    return 0


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="FRD seed-robustness add-on (see module docstring).")
    parser.add_argument("command", choices=["plan", "train", "aigen", "analyze", "all"])
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--dry-run", action="store_true", help="train: run every check, start no training")
    parser.add_argument("--dataset", default="test.csv", help="AiGen-FoodReview official test.csv")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--native-retries", type=int, default=1, choices=(0, 1))
    parser.add_argument("--max-task-hours", type=float, default=0)
    parser.add_argument("--force-precision", action="append", default=[],
                        help="MODEL=bf16|fp16|fp32; overrides the seed-42 precision for new runs (recorded as amendment 2)")
    args = parser.parse_args(argv)
    forced = parse_forced_precision(args.force_precision)
    project = args.project.expanduser().resolve()
    ensure_safe_interpreter(project, argv)

    if args.command == "plan":
        return cmd_plan(project, args.stage)
    if args.command == "analyze":
        return cmd_analyze(project)
    codes = []
    if args.command in ("train", "all"):
        codes.append(cmd_train(project, args.stage, args.native_retries, args.max_task_hours, args.dry_run, forced))
        if args.dry_run:
            return codes[-1]
        if codes[-1] and args.command == "all":
            print("[STOP] Training reported failures; AiGen scoring and analysis were not started.")
            return codes[-1]
    if args.command in ("aigen", "all"):
        codes.append(cmd_aigen(project, args.stage, args.dataset, args.device))
    if args.command == "all":
        codes.append(cmd_analyze(project))
    return max(codes) if codes else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoints are kept; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(3)