# Fake Review Detection across Classical, Deep, and Transformer Models

This repository contains the code, data, predictions, and statistical outputs
for a leakage-controlled comparison of classical machine-learning,
deep-learning, and Transformer detectors for machine-generated reviews, with a
zero-shot evaluation on reviews from a newer generator and domain.

## Overview

Reported fake-review detection results are hard to compare because studies
differ in data partitions, duplicate handling, threshold selection, and use of
the test set during development. This study evaluates 25 classifiers and a
soft-voting ensemble on the Fake Reviews Dataset (FRD; human-written Amazon
reviews versus GPT-2-generated reviews) under one protocol:

- one duplicate-safe 70/10/20 partition shared by all systems
- every development decision (checkpoints, thresholds, models, ensembles) made on validation data only
- a test set used only for final evaluation and paired inference
- a zero-shot test on AiGen-FoodReview (human-written Yelp reviews versus GPT-4-Turbo-generated reviews) with all FRD decisions frozen

## Research Questions

1. How does performance change from traditional to deep-learning to pretrained Transformer representations under a common partition?
2. Are the gains between stages statistically supported on the same test observations?
3. Does sentiment augmentation help consistently across Transformer backbones?
4. Do the FRD-trained systems retain useful discrimination zero-shot on AiGen-FoodReview?

## Main Findings

| Stage | System | FRD test accuracy |
|---|---|---|
| Traditional ML | TF-IDF + SVM | 94.32% |
| Deep learning | Word2Vec + LSTM | 95.00% |
| Plain Transformer | ModernBERT | 98.23% |
| Final ensemble | DistilBERT + S, ModernBERT + S, RoBERTa + S (weights 0.10 / 0.45 / 0.45) | 98.75% (ROC-AUC 0.9991) |

- The largest gain comes with pretrained contextual representations; the ensemble adds a small but significant improvement.
- Sentiment augmentation is architecture-dependent: it improves RoBERTa, degrades BERT and ELECTRA, and shows no significant accuracy difference for the other four backbones.
- On AiGen-FoodReview every system labels nearly all GPT-4-Turbo reviews as authentic. The final ensemble ranks the classes in reverse (ROC-AUC 0.4357), while plain BERT keeps a ROC-AUC of 0.9055.

## Quick Look

No installation is needed to inspect the results:

- all 25 classifiers and the ensembles, with validation and test metrics and full confusion counts: [`results/frd_benchmark/all_systems.csv`](results/frd_benchmark/all_systems.csv)
- paired tests and bootstrap intervals: [`results/statistics/`](results/statistics/)
- zero-shot AiGen-FoodReview results: [`results/external_aigen/all_systems_primary_metrics_with_ci.csv`](results/external_aigen/all_systems_primary_metrics_with_ci.csv)

[`results/README.md`](results/README.md) lists which file gives which table of the paper.

## Repository Structure

```
fake-review-detection-cross-paradigm/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ .gitignore
├─ requirements.txt                 Transformer environment (pinned)
├─ requirements-baselines.txt       baseline environment
├─ data/
│  ├─ README.md                     sources, licenses, label convention
│  ├─ fake reviews dataset.csv      FRD (training, validation, test)
│  ├─ deceptive-opinion.csv         OpSpam (zero-shot test)
│  ├─ test.csv                      AiGen-FoodReview test split (zero-shot test)
│  ├─ frd_split.csv                 FRD partition used by every system
│  └─ licenses/                     license notices of the included datasets
├─ docs/
│  ├─ reproducibility.md            how to run, environments, provenance notes
│  └─ environment/                  full package lists of both environments
├─ results/
│  ├─ README.md                     which file gives which table of the paper
│  ├─ frd_benchmark/                25 classifiers: metrics and validation/test predictions
│  ├─ selection/                    backbone representatives and ensemble search
│  ├─ statistics/                   McNemar tests, Holm correction, bootstrap intervals
│  ├─ external_aigen/               zero-shot AiGen-FoodReview results and diagnostics
│  ├─ external_opspam/              zero-shot OpSpam results
│  └─ checks/                       selection records and the ModernBERT CPU/GPU check
└─ scripts/
   ├─ FRD_Run_Transformer_Study.py
   ├─ FRD_Run_NonTransformer_Study.py
   ├─ FRD_CrossParadigm_Statistical_Analysis.py
   └─ FRD_AiGen_External_Validation.py
```

## Scripts

| Script | Role |
|---|---|
| `FRD_Run_Transformer_Study.py` | Environment setup, duplicate-safe split, training of the 7 plain and 7 sentiment-augmented Transformers, representative selection, ensemble search, OpSpam zero-shot test |
| `FRD_Run_NonTransformer_Study.py` | 6 traditional machine-learning and 5 deep-learning baselines |
| `FRD_CrossParadigm_Statistical_Analysis.py` | Paired McNemar tests with Holm correction and paired bootstrap intervals |
| `FRD_AiGen_External_Validation.py` | Zero-shot AiGen-FoodReview evaluation; `diagnostics` and `device-check` sub-commands |

[`docs/reproducibility.md`](docs/reproducibility.md) gives the order of the
steps and the environments.

## Data

The three datasets are included in [`data/`](data/) with their sources and
licenses; please cite the original authors when using them.

## Citation

The accompanying article is under peer review; its reference will be added
here after publication. Until then, please cite this repository using
`CITATION.cff` ("Cite this repository" on GitHub).

## License

Code and result files: MIT (see `LICENSE`). The datasets keep their original
terms (see [`data/README.md`](data/README.md)).
