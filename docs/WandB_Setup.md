# Experiment Tracking with Weights & Biases

This guide covers how to enable W&B tracking for the borehole extraction pipeline.
Tracking is **optional** — the pipeline runs fine without it.

## Prerequisites

- A free [Weights & Biases](https://wandb.ai) account
- `wandb` and `pygit2` installed

---

## Step 1 — Install tracking dependencies

```bash
uv sync --extra wandb
```

---

## Step 2 — Configure your environment

Open `.env` and set:

```
WANDB_TRACKING=True
WANDB_API_KEY=<your key>       # from https://wandb.ai/authorize
WANDB_PROJECT=swissgeol-boreholes   # optional, this is the default
WANDB_BASE_URL=https://api.wandb.ai # to track experiments in the users UI
MLFLOW_TRACKING=False
```

Then reload your environment:

```bash
source .env
```

---

## Step 3 — Run the pipeline
Pass a ground truth file with `-g` to get evaluation metrics — without it only `n_documents` is logged.

### Single directory

```bash
boreholes-extract-all -i data/pdfs/ -g data/ground_truth.json
```

### Multiple benchmarks

```bash
boreholes-extract-all \
  --benchmark "name1:data/set1/:data/gt1.json" \
  --benchmark "name2:data/set2/:data/gt2.json" \
  -o data/benchmarks/
```

Each `--benchmark` argument follows the format `name:input_directory:ground_truth_path`.
All benchmark runs are grouped together in the W&B UI under a shared group name.

---

## Step 4 — View results

Open your W&B project at `https://wandb.ai/<your-entity>/swissgeol-boreholes`.

| What | Where in W&B |
|---|---|
| Metrics (F1, precision, recall) | Charts panel of each run |
| Config (input path, git commit, YAML params) | Overview tab |
| Prediction images | Media panel (`predictions` key) |
| Benchmark summary JSON | Files tab |
| Multi-benchmark aggregate | The `aggregate` run in the group |

Output files are also written to disk under `--out-directory`:

- `predictions.json` — extracted borehole data
- `metadata.json` — per-file metadata (coordinates, names)
- `benchmark_summary.json` — evaluation metrics (if ground truth was provided)
- `overall_summary.csv` — aggregate metrics across benchmarks (multi-benchmark mode only)

---

## Troubleshooting
**`wandb: ERROR ...` on first run**
Check that `WANDB_API_KEY` is set correctly and authenticate:
```bash
wandb login $WANDB_API_KEY
```

**Find your W&B entity name**
Your entity is not necessarily your username — it may be a team name:
```bash
python -c "import wandb; r = wandb.init(project='test'); print(r.entity); wandb.finish()"
```
