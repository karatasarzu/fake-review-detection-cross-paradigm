# Reproducibility

## Working folder

The scripts read the data files from, and write `results/` to, the folder they
are run in. Create a working folder with the four scripts and the three data
files (PowerShell):

```
git clone https://github.com/karatasarzu/fake-review-detection-cross-paradigm.git
cd fake-review-detection-cross-paradigm
mkdir work
copy scripts\*.py work\
copy "data\fake reviews dataset.csv" work\
copy data\deceptive-opinion.csv work\
copy data\test.csv work\
cd work
```

## Steps

Run the steps in this order. Every step reuses verified outputs, so running a
step again only checks it.

```
# 1. Transformer environment (system Python 3.9-3.12), then all Transformer steps
python FRD_Run_Transformer_Study.py setup
.venv_frd_safe\Scripts\python.exe FRD_Run_Transformer_Study.py all

# 2. Baselines, in a separate environment (reuses the split from step 1)
python -m venv .venv_baselines
.venv_baselines\Scripts\python.exe -m pip install -r ..\requirements-baselines.txt
.venv_baselines\Scripts\python.exe FRD_Run_NonTransformer_Study.py

# 3. Paired tests and bootstrap intervals
.venv_frd_safe\Scripts\python.exe FRD_CrossParadigm_Statistical_Analysis.py

# 4. Zero-shot AiGen-FoodReview evaluation and post-hoc checks
.venv_frd_safe\Scripts\python.exe FRD_AiGen_External_Validation.py --scope all
.venv_frd_safe\Scripts\python.exe FRD_AiGen_External_Validation.py diagnostics --frd-dataset "fake reviews dataset.csv"
.venv_frd_safe\Scripts\python.exe FRD_AiGen_External_Validation.py device-check

# 5. Repeated training runs (seeds 123, 2025, 3407, 7777) and their AiGen scoring
.venv_frd_safe\Scripts\python.exe FRD_Run_Seed_Robustness.py all --stage 1
.venv_frd_safe\Scripts\python.exe FRD_Run_Seed_Robustness.py all --stage 2
.venv_frd_safe\Scripts\python.exe FRD_Run_Seed_Robustness.py all --stage 3 --force-precision deberta=fp32

# 6. Rank correlations between FRD accuracy and AiGen ROC-AUC
.venv_frd_safe\Scripts\python.exe FRD_Seed_Correlations.py
```

Step 5 follows the recorded protocol: stage 1 retrains the three pairs whose
sentiment effect was significant in the first run (BERT, RoBERTa, ELECTRA)
with seeds 123 and 2025; stage 2 adds seeds 3407 and 7777 for the pairs whose
sign changed; stage 3 (protocol amendment 1) completes all seven pairs with
all four seeds. The protocol and both amendments are written to
`results/seed_robustness/` before the runs they cover. Running stage 3 alone
reproduces the full grid, because verified runs are reused.

`FRD_Run_Transformer_Study.py` also accepts `split`, `deberta-v3`, `train`,
and `status` to run or check single steps. The baseline script must run after
the split step, because it reuses the Transformer partition and creates its
own only when none exists. The partition written to `results/splits/` can be
checked against `data/frd_split.csv`.

## Environments

| | Transformer study | Baselines |
|---|---|---|
| Python | 3.9.1 | see `docs/environment/baselines_pip_freeze.txt` |
| Main packages | PyTorch 2.5.1 (CUDA 11.8), Transformers 4.48.3, Accelerate 1.2.1, scikit-learn 1.4.2, TextBlob 0.18.0 | PyTorch 2.1.2 (CUDA 11.8), scikit-learn 1.4.2, gensim 4.3.3, NLTK 3.9.2 |
| Package list | `requirements.txt`, `docs/environment/transformers_pip_freeze.txt` | `requirements-baselines.txt`, `docs/environment/baselines_pip_freeze.txt` |

Hardware: one NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), recorded in
`docs/environment/transformers_environment.json`. The traditional and
deep-learning baselines and the ensembles were trained once, with seed 42.
Each of the 14 Transformer configurations was trained with seed 42 and
retrained with seeds 123, 2025, 3407, and 7777 (70 runs; step 5). The
seed-42 runs are the ones in `results/frd_benchmark/`; the repeated runs are in
`results/seed_robustness/frd/` and `results/external_aigen/seed_<N>/`.

## Structure of `FRD_Run_Transformer_Study.py`

The file contains the process supervisor that runs each training task in an
isolated process and verifies its outputs, the task lists of the study, the
environment setup, a separate routine for plain DeBERTa-v3, and the embedded
runtime (data split, model registry, training loop, metrics, and threshold
rule). The runtime is stored as a zip archive, checked against its SHA-256
before use, and extracted to `.frd_supervisor/runtime/`.

## Provenance notes

- **Plain DeBERTa-v3** was trained in FP32 by a separate routine with the
  checkpoint's SentencePiece tokenizer, and its classification head was
  initialized before the training seed was set. Its validation and test
  predictions were regenerated from the saved weights in FP32
  (`results/frd_benchmark/transformers/deberta_v3_base/reconciliation.json`).
- **Numeric precision.** Transformers were fine-tuned in bfloat16 mixed
  precision, except the DeBERTa sentiment variant (FP32), the gated DeBERTa-v3
  variant (FP16), and plain DeBERTa-v3 (FP32).
- **Repeated runs.** Each repeated run keeps the partition, sentiment
  features and scaler, hyperparameters, checkpoint and threshold rules, and
  numeric precision of the model's seed-42 run; the seed changes the
  initialization of the classification and fusion layers, the batch order, and
  dropout. Two exceptions are recorded in `results/seed_robustness/`:
  plain DeBERTa was retrained in FP32 (bfloat16 training overflowed at the
  first step; protocol amendment 2), and plain DeBERTa-v3 was retrained in
  FP32 with the common training routine rather than the separate seed-42
  routine. The DeBERTa pair is therefore precision-matched only in the
  repeated runs, and the DeBERTa-v3 pair (FP32 plain, FP16 gated) in none.
- **AiGen inference.** Both ModernBERT configurations were scored on CPU in
  FP32 with eager attention, and both DeBERTa-v3 configurations on GPU in FP32.
  Re-scoring the FRD test set on the same CPU path reproduces the stored GPU
  predictions (`results/checks/modernbert_device_check.json`).
- **Evaluation script version.** The reported AiGen run used v1.0.3 of the
  evaluation code (`FRD_AiGen_External_Validation.py`). Predictions of the
  other systems were first produced by earlier versions whose inference path
  for those systems is identical.
- **Not reported.** The Transformer study also runs OpSpam fine-tuning tasks
  and two auxiliary models (sentiment-only, BERT + polarity); they are not used
  in the paper, and their outputs are not included here.
