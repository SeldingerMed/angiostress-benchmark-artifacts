---
license: other
pretty_name: AngioStress Benchmark Artifacts
task_categories:
- image-segmentation
tags:
- angiography
- dsa
- benchmark
- reproducibility
- construct-validity
---

# AngioStress Benchmark Artifacts

This repository is the runnable artifact package for **AngioStress: A deterministic synthetic DSA stress-test benchmark for construct-validity measurement in frozen endovascular perception models**.

AngioStress is not a clinical-validation benchmark and does not claim sim-to-real transfer. It defines deterministic DSA-like stressor cells and measures whether stressor severity predicts the ordering of model failures observed on real angiography surfaces.

Author: Colin Son, MD, Seldinger, Inc., San Antonio, TX. ORCID: https://orcid.org/0000-0002-1782-0537

Public mirrors:

- GitHub: https://github.com/SeldingerMed/angiostress-benchmark-artifacts
- Hugging Face dataset: https://huggingface.co/datasets/txmedai/angiostress-benchmark-artifacts

## Benchmark Cell

A benchmark cell is `(source surface, stressor or real-surface factor, severity/bin, model, prompt policy, metric)`. The package records synthetic stressor cells from a TopCoW-derived DSA-like projection and real-angiography surfaces from DIAS and CathAction. The frozen model panel is SAM ViT-B, MedSAM ViT-B, and SAM ViT-L, with no training or fine-tuning.

## Runnable Surface

The repository uses the same relative layout expected by the scripts:

- `experiments/main/run-angiostress-s0s1-renderer-smoke/run_renderer_smoke.py`: deterministic synthetic renderer scaffold.
- `experiments/main/run-angiostress-s1-frozen-model-smoke/run_sam_smoke.py`: single frozen-model scaffold.
- `experiments/main/run-angiostress-s2c-third-frozen-model-panel-extension/run_model_panel_smoke.py`: three-model synthetic/DIAS panel scaffold.
- `experiments/analysis/s3a-dias-test-multisequence-ranking/run_dias_multisequence_ranking.py`: DIAS labeled sequence diagnostic script.
- `experiments/analysis/s3b-dias-full-test-split-ranking/run_dias_full_test_split.py`: DIAS full labeled test split rerun script generated from the S3A evaluator with S3B defaults.
- `experiments/analysis/s3f-cathaction-human-segmentation-subset-ranking/run_cathaction_human_segmentation_subset.py`: CathAction human-segmentation subset diagnostic script.

Install the Python dependencies in `requirements.txt`. Full model reruns require upstream SAM/MedSAM checkpoints placed at the relative paths recorded in `tmp/angiostress-*` or supplied by editing the model panel paths. Full DIAS/CathAction reruns require the upstream datasets listed in `DATA_SOURCES.md`.

## Example Commands

Renderer scaffold:

```bash
python experiments/main/run-angiostress-s0s1-renderer-smoke/run_renderer_smoke.py   --out experiments/main/run-angiostress-s0s1-renderer-smoke/outputs_rerun
```

Three-model panel scaffold after placing checkpoints and DIAS data at the recorded relative paths:

```bash
python experiments/main/run-angiostress-s2c-third-frozen-model-panel-extension/run_model_panel_smoke.py   --workspace-root .   --out experiments/main/run-angiostress-s2c-third-frozen-model-panel-extension/outputs_rerun   --device auto   --max-synthetic-items 0   --max-dias-frames 0
```

DIAS full test split diagnostic:

```bash
python experiments/analysis/s3b-dias-full-test-split-ranking/run_dias_full_test_split.py   --dias-zip tmp/datasets/DIAS/DIAS.zip   --out experiments/analysis/s3b-dias-full-test-split-ranking/outputs_rerun   --device auto   --no-visuals
```

CathAction human-segmentation subset diagnostic:

```bash
python experiments/analysis/s3f-cathaction-human-segmentation-subset-ranking/run_cathaction_human_segmentation_subset.py   --zip-path tmp/cathaction_s3f/segmentation_human_train.zip   --out experiments/analysis/s3f-cathaction-human-segmentation-subset-ranking/outputs_rerun   --device auto
```

## Included Evidence

The recorded outputs show finite metrics for all evaluated model-surface cells. Construct-validity evidence is mixed: the DIAS full test split is discordant with the synthetic ordering, while the CathAction human-segmentation subset partially agrees by retaining MedSAM ViT-B as the weakest model but swapping SAM ViT-B and SAM ViT-L relative to the synthetic surface.

## Redistribution Boundary

This public package intentionally excludes manuscript source, generated LaTeX, compiled manuscript PDFs, paper review/comment files, upstream raw datasets, and third-party model weights/checkpoints. It includes code, configs, validation records, generated benchmark outputs, compact metrics, and manifests needed to inspect and rerun the benchmark scaffold.

Generated at: 2026-06-15T14:44:21.980274+00:00
