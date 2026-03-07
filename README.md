# SpectralNet

<p align="center">
    <img src="https://github.com/shaham-lab/SpectralNet/blob/main/figures/twomoons.png">

SpectralNet is a Python package that performs spectral clustering with deep neural networks.<br><br>
This package is based on the following paper - [SpectralNet](https://openreview.net/pdf?id=HJ_aoCyRZ)

## Installation

### From PyPI

```bash
pip install spectralnet
```

### From source (with pixi)

[pixi](https://pixi.sh) is the recommended way to set up a fully reproducible
development environment after cloning the repo.

```bash
# 1. Install pixi (once, system-wide)
curl -fsSL https://pixi.sh/install.sh | sh

# 2. Clone and enter the repo
git clone https://github.com/shaham-lab/SpectralNet.git
cd SpectralNet

# 3. Install all dependencies (conda + PyPI) into an isolated environment
pixi install

# 4. Run the test suite to verify everything works
pixi run test
```

After `pixi install` you can prefix any command with `pixi run` to execute it
inside the managed environment, or activate the environment with:

```bash
pixi shell
```

## Usage

### Clustering — small datasets (in-memory tensor)

For datasets that fit in RAM, pass a `torch.Tensor` directly:

```python
from spectralnet import SpectralNet

spectralnet = SpectralNet(n_clusters=10)
spectralnet.fit(X)                          # X: torch.Tensor of shape (N, ...)
cluster_assignments = spectralnet.predict(X)
```

To measure ACC and NMI when labels are available:

```python
from spectralnet import SpectralNet, Metrics

spectralnet = SpectralNet(n_clusters=2)
spectralnet.fit(X, y)                       # y: integer label tensor
cluster_assignments = spectralnet.predict(X)

y_np = y.detach().cpu().numpy()
acc_score = Metrics.acc_score(cluster_assignments, y_np, n_clusters=2)
nmi_score = Metrics.nmi_score(cluster_assignments, y_np)
print(f"ACC: {acc_score:.3f}  NMI: {nmi_score:.3f}")
```

### Clustering — large datasets (streaming from disk)

For datasets too large to hold in RAM (e.g. millions of images on disk),
define a `torch.utils.data.Dataset` that loads **one sample at a time**
and pass it to `fit()`. Nothing large ever lives in memory at once — every
trainer pulls mini-batches through its own `DataLoader` internally.

```python
from torch.utils.data import Dataset, DataLoader
from spectralnet import SpectralNet
from PIL import Image
import torchvision.transforms as T
import os

class ImageFolderDataset(Dataset):
    def __init__(self, root):
        self.paths = [
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".jpg")
        ]
        self.transform = T.Compose([T.Resize(64), T.ToTensor(), T.Normalize(0.5, 0.5)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))

dataset = ImageFolderDataset("/path/to/images")

spectralnet = SpectralNet(
    n_clusters=10,
    should_use_ae=True,              # compress images before clustering
    ae_hiddens=[2048, 512, 64, 10],
    spectral_hiddens=[512, 512, 10],
)
spectralnet.fit(dataset)

# predict() also accepts a DataLoader for large test sets
test_loader = DataLoader(dataset, batch_size=512, shuffle=False)
cluster_assignments = spectralnet.predict(test_loader)
```

> **Note on Siamese training with large datasets:** the Siamese network
> builds exact k-NN pairs, which requires loading all features into memory.
> For very large datasets either disable it (`should_use_siamese=False`),
> enable approximate neighbours (`siamese_use_approx=True`), or pass a
> representative subset as the Dataset.

### Running examples

```bash
cd examples
python3 cluster_twomoons.py
python3 cluster_mnist.py
```

<!-- ### Data reduction and visualization

SpectralNet can also be used as an effective and representive data reduction technique, so in case you want to perform data reduction you can do the following:

```python
from spectralnet import SpectralReduction

spectralreduction = SpectralReduction(
    n_components=3,
    should_use_ae=True,
    should_use_siamese=True,
)

X_new = spectralreduction.fit_transform(X)
spectralreduction.visualize(X_new, y, n_components=2) -->

<!-- ``` -->

## Citation

```

@inproceedings{shaham2018,
author = {Uri Shaham and Kelly Stanton and Henri Li and Boaz Nadler and Ronen Basri and Yuval Kluger},
title = {SpectralNet: Spectral Clustering Using Deep Neural Networks},
booktitle = {Proc. ICLR 2018},
year = {2018}
}

```
