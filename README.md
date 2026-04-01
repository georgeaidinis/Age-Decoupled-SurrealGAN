# Age-Decoupled SurrealGAN

This repository is a substantial rebuild of the original [SurrealGAN](https://github.com/zhijian-yang/SurrealGAN) codebase from [Zhijian Yang](https://zhijian-yang.github.io/), adapted for a new research goal: **explicitly decoupling aging from other brain processes in ROI space**.

The original `SurrealGAN/` package is still present as historical reference. The new pipeline lives under [`src/age_decoupled_surrealgan`](src/age_decoupled_surrealgan) and is designed as a full research stack:

- config-driven preprocessing and dataset splitting
- age-decoupled ROI model training
- repeated-run agreement scoring
- TensorBoard logging and structured run artifacts
- Optuna-based hyperparameter tuning
- FastAPI backend for inference
- React/Niivue web GUI for interactive visualization
- notebook-based dataset EDA

## Scientific framing

The new model keeps the broad SurrealGAN idea of learning transformations in ROI space, but changes the latent structure:

- one explicit **age latent**
- `K` non-age **process latents** (`r1..rK`)

_The intent is to let the age latent capture age-related structure while the process latents uncover additional patterns that may correspond to disease or other mechanisms._

## Environment

Create the new Conda environment you requested:

```bash
conda create -n age-decoupled-surrealgan python=3.11 -y
conda activate age-decoupled-surrealgan

pip install --upgrade pip setuptools wheel
pip install -e .
pip install -e ".[dev]"

# Frontend runtime
conda install -c conda-forge nodejs -y
```

If you want to install the dependencies explicitly instead of using `pip install -e .`, the intended stack is:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scipy scikit-learn tensorboard matplotlib nibabel nilearn
pip install fastapi uvicorn[standard] pydantic optuna
pip install pytest pytest-cov ruff black mypy
```

## Dataset assets

The repo expects these local assets in [`datasets/`](datasets):

- `cleaned_istaging.csv`
- `MUSE_ROI_complete_list.json`
- `atlas.nii.gz`
- `atlas_segmentation.nii.gz`

The atlas as provided is the skull-stripped IXI subject `IXI016-Guys-0697-T1.nii.gz`, segmented with DLMUSE, and is used for visualization only. This can be any atlas and segmentation `.nii.gz` files.

Only **atomic** MUSE ROI volumetrics are used as model inputs. Any feature whose column name ends in an ROI ID above `299` is excluded during preprocessing, so composite regions such as `MUSE_Volume_345` are dropped everywhere in the pipeline.

## Default split policy

The preprocessing command creates a deterministic split manifest with the following defaults:

- REF cohort: cognitively normal only, ages `20-49`
- TAR cohort: all subjects, ages `50-97`
- OOD/application holdout: whole study `HANDLS`
- Remaining studies: subject-level `train / val / id_test = 80 / 10 / 10`

Generated artifacts are written under [`artifacts/data/processed`](artifacts/data/processed):

- `train.csv`
- `val.csv`
- `id_test.csv`
- `ood_test.csv`
- `application.csv`
- `all_rows.csv`
- `eligible_rows.csv`
- `roi_metadata.csv`
- `reference_template.csv`
- `split_manifest.json`
- `split_subject_ids.json`

## Notebook EDA

The dataset exploration notebook is at:

- [`output/jupyter-notebook/cleaned-istaging-eda-and-split-design.ipynb`](output/jupyter-notebook/cleaned-istaging-eda-and-split-design.ipynb)

It documents:

- study distribution
- REF/TAR cohort logic
- the `HANDLS` OOD holdout choice
- generated split artifacts
- ROI metadata and atlas inputs

## Documentation

Detailed documentation lives in [`docs/`](docs):

- [`docs/objectives_and_losses.md`](docs/objectives_and_losses.md): original vs extension objectives, mathematical formulations, and ablation options
- [`docs/metrics_and_logging.md`](docs/metrics_and_logging.md): metric definitions, TensorBoard organization, terminal logs, and text-readable outputs
- [`docs/pipeline_usage.md`](docs/pipeline_usage.md): command-line usage and experiment workflow
- [`docs/slurm_workflows.md`](docs/slurm_workflows.md): cluster submission guidance and script usage

## CLI workflow

Run commands from the repo root.

Prepare processed artifacts:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli prepare-data
```

Train the default configuration:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli train
```

Run a quick smoke config:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli --config src/age_decoupled_surrealgan/configs/quickstart.toml train
```

Run Optuna tuning:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli tune
```

Start the API:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli serve
```

## Configuration

The default config is:

- [`src/age_decoupled_surrealgan/configs/default.toml`](src/age_decoupled_surrealgan/configs/default.toml)

Important config sections:

- `paths`: raw data, atlas files, processed outputs, run outputs
- `data`: cohort thresholds, holdout study, split seed, and the atomic-ROI cutoff `max_atomic_roi_id = 299`
- `model`: number of process latents and hidden widths
- `training`: epochs, repetitions, batch size, device, checkpoint cadence
- `losses`: weighting for age supervision, age adversary, decomposition, identity, covariance, orthogonality
- `tuning`: Optuna trial count
- `app`: API host/port

## Training outputs

Each training run writes a timestamped directory under [`runs/`](runs) with:

- `resolved_config.json`
- `run_summary.json`
- `metrics/run_summary.md`
- per-repetition checkpoints
- split-level latent prediction CSVs
- split-level metrics JSONs
- `metrics/split_metrics.csv`
- `metrics/epoch_history.csv`
- `logs/train.log`
- `logs/epoch_history.jsonl`
- TensorBoard event files

`run_summary.json` contains:

- the selected repetition
- the selected checkpoint path
- mean agreement across repetitions
- per-split metrics for `train`, `val`, `id_test`, `ood_test`, and `application`

TensorBoard groups scalars by objective family and validation metric family, and embeds the objective and metric explainers directly under `docs/*`.

## Web GUI

The frontend lives in [`webui/`](webui). It is a Vite/React application with:

- dynamic slider count inferred from checkpoint metadata
- subject selection from processed splits
- 2D multiplanar slice view by default
- optional 3D volume mode
- ROI change bar chart
- Niivue-based loading of the atlas and segmentation

Install frontend dependencies and run it with:

```bash
cd webui
npm install
npm run dev
```

The Niivue dependency is installed from the scoped npm package `@niivue/niivue`.

You will typically run the API on `127.0.0.1:8000` and the UI on `127.0.0.1:5173`.

## SLURM

Example SLURM scripts for CUBIC-style cluster usage are provided in [`scripts/slurm/`](scripts/slurm):

- [`scripts/slurm/prepare_data.slurm`](scripts/slurm/prepare_data.slurm)
- [`scripts/slurm/train.slurm`](scripts/slurm/train.slurm)
- [`scripts/slurm/tune.slurm`](scripts/slurm/tune.slurm)
- [`scripts/slurm/train_array.slurm`](scripts/slurm/train_array.slurm)

## Code layout

- [`src/age_decoupled_surrealgan/config.py`](src/age_decoupled_surrealgan/config.py): typed project config
- [`src/age_decoupled_surrealgan/data/prepare.py`](src/age_decoupled_surrealgan/data/prepare.py): preprocessing and split generation
- [`src/age_decoupled_surrealgan/model.py`](src/age_decoupled_surrealgan/model.py): age-decoupled ROI model
- [`src/age_decoupled_surrealgan/trainer.py`](src/age_decoupled_surrealgan/trainer.py): training loop and run orchestration
- [`src/age_decoupled_surrealgan/metrics.py`](src/age_decoupled_surrealgan/metrics.py): agreement and age-leakage metrics
- [`src/age_decoupled_surrealgan/api/app.py`](src/age_decoupled_surrealgan/api/app.py): FastAPI backend
- [`webui/src/App.tsx`](webui/src/App.tsx): frontend app shell

## Current status

Implemented in this repo:

- new packaging and config layer
- preprocessing and real split artifacts for `cleaned_istaging.csv`
- exclusion of composite MUSE ROI features above ID `299`
- ROI metadata export from `MUSE_ROI_complete_list.json`
- age-decoupled model scaffold and training loop
- repeated-run agreement scoring
- tuning scaffold
- FastAPI serving layer
- React/Niivue GUI scaffold
- notebook-based dataset EDA
- README refresh

Still expected from the new environment at runtime:

- installation of the newer Python/web stack in `age-decoupled-surrealgan`
- `npm install` inside `webui/`
- fuller empirical iteration on loss weighting and model behavior once training begins

## Citation

If you use the original SurrealGAN method, cite the original work:

```bibtex
@inproceedings{yang2022surrealgan,
  title={Surreal-{GAN}: Semi-Supervised Representation Learning via {GAN} for uncovering heterogeneous disease-related imaging patterns},
  author={Zhijian Yang and Junhao Wen and Christos Davatzikos},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=nf3A0WZsXS5}
}
```
