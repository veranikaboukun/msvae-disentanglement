# Disentanglement of Sources in a Multi-Stream Variational Autoencoder

## Overview 
The [pretraining](./pretraining), [training](./training) and [evaluation](./evaluation) directories and corresponding sub-directories contain implementations of Multi-Stream Variational Autoencoder (MS-VAE) to be presented at ECML PKDD 2026.

The [data](./data) directory contains the necessary data files which were used in section 3.2 for MNIST digit source separation. The raw audio files from the [database](https://zenodo.org/records/17342367) used in section 3.3 cannot be provided due to data protection regulations. 

Please follow the [Setup](#setup) instructions described below to run the experiments. 

The code has only been tested on Linux systems.

## Setup
We recommend [Anaconda](https://www.anaconda.com/) to manage the installation, and to create a new environment for hosting the installed packages:

```bash
conda create -n msvae python==3.8.10 
conda activate msvae
```

Install the required packages via pip:

```bash
pip install -r requirements.txt
```

## Reference

```bibtex
@article{
    title={{Disentanglement of Sources in a Multi-Stream Variational Autoencoder}},
    author={Boukun, Veranika and L{\"u}cke, J{\"o}rg},
    year={2025},
    pages = {DOI:10.48550/arXiv.2510.15669},
}
```
