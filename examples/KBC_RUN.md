# SpectralNet on KBC datasets

`cluster_kbc.py` trains and evaluates SpectralNet. `predict_kbc.py` reloads a
saved model. Both scripts resolve paths from their own location, so they can be
launched from any working directory.

## Paths

- Input datasets: `../data`, relative to the repository root.
- Run artifacts: `examples/results/<dataset>/`.
- Consolidated metrics and run parameters: `examples/results/results.csv`.
- Override the input root with `--data_root`; result locations are fixed to keep
  every dataset's artifacts together.

For example:

```bash
python examples/cluster_kbc.py --key COIL20 --use_ae --use_siamese --save_model
python examples/predict_kbc.py --key COIL20 --eval
```

Each result directory may contain:

```text
examples/results/COIL20/
├── COIL20_result.csv
├── COIL20_model.pt
├── COIL20_scaler.pkl
├── COIL20_predictions.npy
├── ae_weights.pth
├── siamese_weights.pth
├── COIL20_dataset.jpg
└── COIL20_clustering_result.jpg
```

The model is saved only when `--save_model` is supplied. The scaler, CSV, and
plots are written by training; predictions are written by the prediction
script. Re-running a dataset replaces artifacts with the same names.

All Python, NumPy, PyTorch CPU/CUDA, scikit-learn, and Annoy random sources use
the fixed seed `42`. cuDNN deterministic mode is enabled and benchmarking is
disabled.

The consolidated CSV keeps the curated dataset order. Metrics and runtime use
four decimal places; learning rates use ordinary decimal notation. A blank
metric or parameter field means that no historical run is available for that
dataset.

## Common training options

| Option | Default |
| --- | --- |
| `--use_ae` | disabled |
| `--use_siamese` | disabled |
| `--ae_hiddens` | `512 256 64` |
| `--ae_epochs` | `30` |
| `--siamese_hiddens` | `512 512 64` |
| `--siamese_epochs` | `20` |
| `--siamese_n_nbg` | `5` |
| `--spectral_hiddens` | `512 512 <n_clusters>` |
| `--spectral_epochs` | `30` |
| `--spectral_lr` | `1e-3` |
| `--spectral_batch_size` | `1024` |
| `--spectral_n_nbg` | `30` |
| `--spectral_scale_k` | `15` |
| `--spectral_is_local_scale` | enabled |

Use `--no-spectral_is_local_scale` to select global scaling.

## Supported datasets

The canonical keys are:

```text
spiral
AC
4C
RingG
complex9
USPS
STL-10
Cifar-10
ImageNet-10
ImageNet-Dogs
MNIST
COIL20
w1Gaussians
w100Gaussians
sparse_3_dense_3_dense_3
sparse_8_dense_1_dense_1
one_gaussian_10_one_line_5_2
tutorial
tonsil
airway
crohn
dlpfc_151507
non_spherical
non_spherical_gap
non_spherical_gap_0_5
non_spherical_gap_0_8
```

`MNIST` maps to `../data/mnist.mat`. `dlpfc_151507` maps to
`../data/stdata/DLPFC_FINAL_PKL/151507_final.pkl` and applies the same spatial
feature propagation during training and prediction.
