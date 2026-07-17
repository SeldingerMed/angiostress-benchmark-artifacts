# Running AngioStress

Install dependencies from `requirements.txt`, place upstream data and checkpoints at the relative paths recorded in `DATA_SOURCES.md` and the run manifests, then execute from the repository root.

The synthetic renderer can be rerun without DIAS or CathAction. The frozen model panel and real-surface diagnostics require public model checkpoints and upstream datasets.

## Minimal synthetic renderer check

```bash
python3 experiments/main/run-angiostress-s0s1-renderer-smoke/run_renderer_smoke.py \
  --config experiments/main/run-angiostress-s0s1-renderer-smoke/config.json \
  --out experiments/main/run-angiostress-s0s1-renderer-smoke/outputs_rerun \
  --synthetic-fixture
```

`--synthetic-fixture` writes a tiny generated NIfTI mask under `outputs_rerun/`; it
does not require the untracked TopCoW source paths recorded in the default config.

## Frozen model panel

```bash
python experiments/main/run-angiostress-s2c-third-frozen-model-panel-extension/run_model_panel_smoke.py   --workspace-root .   --out experiments/main/run-angiostress-s2c-third-frozen-model-panel-extension/outputs_rerun   --device auto   --max-synthetic-items 0   --max-dias-frames 0
```

## DIAS full labeled test split

```bash
python experiments/analysis/s3b-dias-full-test-split-ranking/run_dias_full_test_split.py   --dias-zip tmp/datasets/DIAS/DIAS.zip   --out experiments/analysis/s3b-dias-full-test-split-ranking/outputs_rerun   --device auto   --no-visuals
```

## CathAction human-segmentation subset

```bash
python experiments/analysis/s3f-cathaction-human-segmentation-subset-ranking/run_cathaction_human_segmentation_subset.py   --zip-path tmp/cathaction_s3f/segmentation_human_train.zip   --out experiments/analysis/s3f-cathaction-human-segmentation-subset-ranking/outputs_rerun   --device auto
```
