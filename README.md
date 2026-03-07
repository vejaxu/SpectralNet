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

### Clustering

The basic functionality is quite intuitive and easy to use, e.g.,

```python
from spectralnet import SpectralNet

spectralnet = SpectralNet(n_clusters=10)
spectralnet.fit(X) # X is the dataset and it should be a torch.Tensor
cluster_assignments = spectralnet.predict(X) # Get the final assignments to clusters
```

If you have labels to your dataset and you want to measure ACC and NMI you can do the following:

```python
from spectralnet import SpectralNet
from spectralnet import Metrics


spectralnet = SpectralNet(n_clusters=2)
spectralnet.fit(X, y) # X is the dataset and it should be a torch.Tensor
cluster_assignments = spectralnet.predict(X) # Get the final assignments to clusters

y = y_train.detach().cpu().numpy() # In case your labels are of torch.Tensor type.
acc_score = Metrics.acc_score(cluster_assignments, y, n_clusters=2)
nmi_score = Metrics.nmi_score(cluster_assignments, y)
print(f"ACC: {np.round(acc_score, 3)}")
print(f"NMI: {np.round(nmi_score, 3)}")
```

You can read the code docs for more information and functionalities<br>

#### Running examples

In order to run the model on twomoons or MNIST datasets, you should first cd to the examples folder and then run:<br>
`python3 cluster_twomoons.py`<br>
or<br>
`python3 cluster_mnist.py`

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
