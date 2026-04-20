# SpectralNet on KBC Datasets

This directory contains scripts to run SpectralNet clustering on KBC-format datasets, including data loading, model training, prediction, and result saving.

## Scripts

| Script | Purpose |
|--------|---------|
| `cluster_kbc.py` | Train SpectralNet on a KBC dataset |
| `predict_kbc.py` | Load a saved model and predict on the same dataset |
| `data.py` | Data loading utilities for all supported KBC datasets |

---

## 1. Training: `cluster_kbc.py`

### Basic Usage

```bash
python cluster_kbc.py --key pendigits
```

### Save Model for Later Prediction

```bash
python cluster_kbc.py --key pendigits --save_model
```

This saves two files under `results/{key}/`:
- `{key}_model.pt` — the complete trained SpectralNet object
- `{key}_scaler.pkl` — the fitted MinMaxScaler used during training

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--key` | required | Dataset name (e.g. `pendigits`, `YaleB`, `COIL20`, `airway`) |
| `--data_root` | `../data` | Root directory of your datasets |
| `--save_model` | `False` | Save trained model and scaler for later prediction |
| `--use_ae` | `False` | Enable AutoEncoder preprocessing |
| `--use_siamese` | `False` | Enable Siamese network preprocessing |
| `--spectral_hiddens` | `[512, 512, n_clusters]` | SpectralNet layer dimensions (last must equal `n_clusters`) |
| `--spectral_epochs` | `30` | Number of training epochs |
| `--spectral_lr` | `1e-3` | Learning rate |
| `--spectral_batch_size` | `1024` | Batch size for SpectralNet training |
| `--ae_hiddens` | `[512, 256, 64]` | AutoEncoder hidden layer dimensions |
| `--siamese_hiddens` | `[512, 512, 64]` | Siamese network hidden layer dimensions |

### Examples

```bash
# Run with default settings
python cluster_kbc.py --key COIL20

# Enable AE + Siamese (recommended for high-dimensional data)
python cluster_kbc.py --key reuters --use_ae --use_siamese --save_model

# Custom network architecture
python cluster_kbc.py --key YaleB --spectral_hiddens 1024 512 38

# Spatial transcriptomics dataset (has special preprocessing)
python cluster_kbc.py --key 151507_final --save_model
```

### Output Structure

After running, the following files are created under `results/{key}/`:

```
results/pendigits/
├── pendigits_result.csv          # NMI, ARI, F1, time
├── pendigits_model.pt            # (if --save_model) trained model
└── pendigits_scaler.pkl          # (if --save_model) fitted scaler
```

And visualization images under `fig/{key}/`:

```
fig/pendigits/
├── pendigits_dataset.jpg         # ground truth scatter plot
└── pendigits_clustering_result.jpg  # clustering result scatter plot
```

---

## 2. Prediction: `predict_kbc.py`

Load a previously saved model and predict cluster assignments. You **must** have run `cluster_kbc.py --save_model` first.

### Basic Usage

```bash
python predict_kbc.py --key pendigits
```

This automatically loads `results/pendigits/pendigits_model.pt` and `results/pendigits/pendigits_scaler.pkl`.

### Evaluate Predictions

```bash
python predict_kbc.py --key pendigits --eval
```

Prints NMI, ARI, and F1 scores against the ground-truth labels.

### Custom Paths

```bash
python predict_kbc.py --key pendigits \
    --model_path /path/to/custom_model.pt \
    --scaler_path /path/to/custom_scaler.pkl
```

### Save Predictions to Custom File

```bash
python predict_kbc.py --key pendigits --output my_predictions.npy
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--key` | required | Dataset name (same as training) |
| `--model_path` | `results/{key}/{key}_model.pt` | Path to saved model |
| `--scaler_path` | `results/{key}/{key}_scaler.pkl` | Path to saved scaler |
| `--data_root` | `../data` | Root directory of your datasets |
| `--eval` | `False` | Evaluate against ground truth |
| `--output` | `results/{key}/{key}_predictions.npy` | Output path for cluster assignments |

---

## Supported Datasets

The following dataset keys are supported out of the box:

### Single-cell / Spatial
- `airway`, `crohn`, `tonsil`, `tutorial` (`.pkl`)
- `151507_final` (`.pkl`, with special WL feature extraction)

### Classic clustering benchmarks (`.mat`)
- `pendigits`, `YaleB`, `reuters`, `landsat`, `waveform3`
- `cure-t2-4k`, `COIL20`, `abalone`, `drybean`, `letters`, `skin`

### Synthetic
- `non_spherical`, `non_spherical_gap`, `non_spherical_gap_0_5`, `non_spherical_gap_0_8`
- `w1Gaussians` through `w1000Gaussians`

To add a new dataset, update the path and variable-name mappings in `data.py` (`get_kbc_mat_file` and `get_kbc_xy_keys`).
