# -*- coding: utf-8 -*-
"""Rank correlations between FRD test accuracy and zero-shot AiGen ROC-AUC.

Reads the per-run tables written by FRD_Run_Seed_Robustness.py (analyze step)
and reports the three Spearman correlations of Section V-C:

  first_run      14 configurations, seed 42 only
  all_runs       all 70 runs (14 configurations x 5 seeds)
  run_averaged   14 configurations, both quantities averaged over the 5 seeds

The correlations are descriptive: configurations share backbones and training
data, so no p-values are reported.

Usage (from the project or repository root):
    python scripts/FRD_Seed_Correlations.py [--results results]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def load(results: Path) -> pd.DataFrame:
    sr = results / "seed_robustness"
    frd = pd.read_csv(sr / "per_seed_model_metrics.csv")[["seed", "model", "test_accuracy"]]
    aig = pd.read_csv(sr / "aigen_per_seed.csv")[["seed", "model", "roc_auc"]]
    df = frd.merge(aig, on=["seed", "model"], how="inner", validate="one_to_one")
    n_models, n_seeds = df.model.nunique(), df.seed.nunique()
    if len(df) != n_models * n_seeds:
        raise SystemExit(f"Incomplete grid: {len(df)} runs for {n_models} models x {n_seeds} seeds.")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    df = load(args.results)

    first = df[df.seed == 42]
    avg = df.groupby("model", as_index=False)[["test_accuracy", "roc_auc"]].mean()
    rows = []
    for name, d in (("first_run", first), ("all_runs", df), ("run_averaged", avg)):
        rho = spearmanr(d.test_accuracy, d.roc_auc).correlation
        rows.append({"scope": name, "n": len(d), "spearman_rho": rho})
    out = pd.DataFrame(rows)
    path = args.results / "seed_robustness" / "spearman_frd_vs_aigen.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Written:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
