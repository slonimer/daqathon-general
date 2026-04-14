# DAQathon Session 1

This repository contains the participant materials for DAQathon Session 1 at Ocean Networks Canada.

The two main notebooks are:

- `notebooks/daqathon_session1_supervised_unsupervised_qc.ipynb`
- `notebooks/daqathon_session1_supervised_unsupervised_qc_tuning.ipynb`

The first notebook is the main teaching notebook. The second is the advanced notebook for tuning, feature engineering, and deeper model comparisons.

## Quick Start On FIR

### 1. SSH to FIR

```bash
ssh <username>@fir.alliancecan.ca
```

### 2. Clone the repo into your `HOME` space

Because this repository is public on GitHub, HTTPS clone is the simplest option on FIR:

```bash
cd "$HOME"
git clone https://github.com/Spiffical/daqathon.git daqathon
cd daqathon
```

If you prefer GitHub SSH and already have your GitHub SSH key configured on FIR, use:

```bash
cd "$HOME"
git clone git@github.com:Spiffical/daqathon.git daqathon
cd daqathon
```

### 3. Launch JupyterHub on FIR

Open:

```text
https://jupyterhub.fir.alliancecan.ca/
```

Start your JupyterHub environment there.

### 4. Open the notebooks from your cloned repo

In the JupyterHub file browser:

- navigate to your cloned `~/daqathon` repo
- open one of the notebooks in `notebooks/`
- select the shared `Daqathon ML` kernel when prompted

Launch the notebooks from inside the cloned repo so the helper imports work correctly.

## What Lives Where

### In your cloned repo in `HOME`

This repo contains:

- the two notebooks
- plotting and lesson assets
- the runtime helper scripts used by the notebooks
- the environment specification

### In shared project space on FIR

These shared resources live under:

```text
/project/def-kmoran/shared/daqathon/
```

The notebooks read shared resources from there:

- shared raw data:
  - `/project/def-kmoran/shared/daqathon/data/raw/...`
- shared prepared cache:
  - `/project/def-kmoran/shared/daqathon/data/cache/...`
- shared virtual environment:
  - `/project/def-kmoran/shared/daqathon/envs/daqathon-ml-venv/`
- shared kernel:
  - `/project/def-kmoran/shared/daqathon/kernels/daqathon-ml/`

Participants should treat the shared data and cache as read-only defaults.

## Which Notebook Should I Use?

### Main notebook

Use `notebooks/daqathon_session1_supervised_unsupervised_qc.ipynb` when you want the guided Session 1 workflow:

- dataset orientation
- QC flags in context
- raw CSV to parquet preparation
- Random Forest baseline
- k-means clustering
- baseline CNN
- transformer introduction

### Advanced notebook

Use `notebooks/daqathon_session1_supervised_unsupervised_qc_tuning.ipynb` when you want to go further:

- hyperparameter search
- feature engineering
- stronger tree models
- alternative target definitions
- sequence-labeling CNN
- deeper neural-network experiments

## Where Outputs Go

The notebooks never write trained models or generated files back into the shared FIR data/cache.

Instead, notebook outputs go to a writable runtime directory chosen in this order:

1. `$SLURM_TMPDIR/daqathon/session1_outputs`
2. `$SCRATCH/daqathon/session1_outputs`
3. `tmp/session1_outputs` inside the cloned repo when running locally

That runtime output area is used for:

- trained model checkpoints
- notebook-generated parquet/cache files
- CSV summaries
- temporary inference/demo outputs
- saved plots

## Shared Cache Behavior

On FIR, the notebooks treat the shared project space as the long-lived source of truth, but they stage fast local working copies when possible.

That means:

- if `$SLURM_TMPDIR` is available, the notebook copies the shared raw CSV directory and the shared prepared cache into node-local job storage before reading them
- shared raw data and shared cache stay as the long-lived sources
- your notebook run gets a fast writable per-run working area
- generated files do not pollute shared project storage

If you choose to run the raw-to-parquet preparation step from inside a notebook, the generated cache will be written into your runtime output directory, not into the shared cache.

## Runtime Files In This Repo

The notebooks import the following runtime helpers:

- `scripts/prepare_ctd_session1_data.py`
- `scripts/session1_modeling.py`

You do not need the local notebook-builder or local experiment-study scripts for the participant workflow.

## Local Development

If you run this repo locally instead of on FIR:

- the notebooks will still work
- they will fall back to repo-local runtime output under `tmp/session1_outputs`
- if you have a local raw-data mirror and local cache, the notebooks can use those too

## Repo Layout

```text
DAQathon/
├── README.md
├── assets/
├── envs/
├── notebooks/
└── scripts/
```

## Notes

- Start Jupyter from the cloned repo root.
- Use the shared FIR kernel for the workshop notebooks.
- Keep large datasets in shared project storage rather than copying them into your own repo clone.
