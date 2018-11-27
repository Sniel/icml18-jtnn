# Junction Tree Variational Autoencoder for Molecular Graph Generation

<img src="https://github.com/wengong-jin/icml18-jtnn/blob/master/paradigm.png" width="400">

Implementation of our Junction Tree Variational Autoencoder [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364)

# Requirements
* Linux (We only tested on Ubuntu)
* RDKit (version >= 2017.09)
* Python (version >= 2.7)
* PyTorch (version >= 0.2)

To install RDKit, please follow the instructions here [http://www.rdkit.org/docs/Install.html](http://www.rdkit.org/docs/Install.html)

We highly recommend you to use conda for package management.

# Quick Start
This repository contains the following directories:
* `bo/` includes scripts for Bayesian optimization experiments. Please read `bo/README.md` for details.
* `molvae/` includes scripts for training our VAE model only. Please read `molvae/README.md` for training our VAE model.
* `molopt/` includes scripts for jointly training our VAE and property predictors. Please read `molopt/README.md` for details.
* `jtnn/` contains codes for model formulation.


# Running on ComputeCanada clusters

That's how I made it work.  There may be other ways that doesn't involve installing miniconda.

1. Install miniconda for Python 3
    eb Miniconda3-4.3.27.eb

2. Create conda env

3. Conda install the following packages:
    RDKit
    Pytorch
    Scipy

4. Run job with one of the .sh scripts:
    molvae_pretrain.sh
    molvae_reconstruct.sh
    ...



