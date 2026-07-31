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
| Prediction images | Files tab | media folder | images folder
| Prediction csv | Files tab | media folder | csv folder
| Benchmark summary JSON | Files tab |

---

## Step 5 — Live prediction comparison table (optional)

Every extraction run logs its own `png_browser_table` (one row per PNG). Right after, it also
rebuilds `prediction_comparison_table`: one image column per run that currently exists in the
project (named after that run's id, including the run that just built it), row-per-file. No
special run, no config, no manual script — this always runs when `WANDB_TRACKING=True`, logged
onto the regular extraction run itself.

### Viewing it

1. Open the **Workspace** tab for your project.
2. Click **"Add panels"**, choose a **Table** panel, and point it at the
   `prediction_comparison_table` key on whichever run you want to inspect (it's logged onto every
   extraction run, so the most recent one has the most columns).
3. Use the panel's own column visibility toggle to show only the run(s) you currently care about.

### Day to day

- Every extraction run automatically rebuilds the table from scratch across every run that
  currently exists in the project and relogs it onto itself.
- Rows are a union across all included runs: a file only some runs processed still gets a row,
  with a blank cell for the runs that didn't process it.
- This scans every run in the project on every extraction run, which adds some time to the end of
  each run as the project's run history grows.

**Caveats:**
- The table only includes files present in *every* included run. Comparing runs over different
  datasets can silently shrink it to nothing — keep only comparable runs visible.
- It scans every finished run in the project on each extraction run to check which ones are
  hidden; on a project with a long run history this adds some time to the end of each run.

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
