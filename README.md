# Disentanglement of Sources in a Multi-Stream Variational Autoencoder

## Overview 
The [pretraining](./pretraining), [training](./training) and [evaluation](./evaluation) directories and corresponding sub-directories contain implementations of Multi-Stream Variational Autoencoder (MS-VAE) to be presented at ECML PKDD 2026.

The MNIST datasets which were used in section 3.2 for MNIST digit source separation can be found here. The raw audio files from the [database](https://zenodo.org/records/17342367) used in section 3.3 cannot be provided due to data protection regulations. 

## Pretrained Models & Requirements

To replicate the results in Table 1, download the following pretrained weights and external files from the original repositories:

### 1. VAE-BSS Checkpoints
* **Files:** `model_vae_K2.pt`, `model_vae_K3.pt`, and `model_vae_K4.pt`
* **Source:** Download from the [VAE-BSS Repository](https://github.com/jundsp/VAE-BSS/tree/main/saves/pretrained).
* **Placement:** Place these files inside your local directory under `pretrained_model_files/`.

### 2. BASIS NCSN Checkpoints & Architecture
* **Weights:** Download `mnist/checkpoint.pth` from the [Basis Separation Repository](https://github.com/jthickstun/basis-separation/tree/master).
* **Architecture:** Download `cond_refinenet_dilated.py` from [ncsn/models/](https://github.com/jthickstun/basis-separation/blob/master/ncsn/models/cond_refinenet_dilated.py).

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
