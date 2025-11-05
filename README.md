# Disentanglement of Sources in a Multi-Stream Variational Autoencoder

## Overview 
The [pretraining](./pretraining), [training](./training) and [evaluation](./evaluation) directories and corresponding sub-directories contain implementations of the experiments described in the [archive paper](https://arxiv.org/abs/2510.15669) submitted to ICASSP 2026. 

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

You will also need to download MNIST dataset, e.g., using torchvision: 

```bash
import torchvision.datasets as datasets
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
```

The raw audio files from the [database](https://zenodo.org/records/17342367) used in this work cannot be provided due to data protection regulations. 

## Reference

```bibtex
@article{
    title={{Disentanglement of Sources in a Multi-Stream Variational Autoencoder}},
    author={Boukun, Veranika and L{\"u}cke, J{\"o}rg},
    year={2025},
    pages = {DOI:10.48550/arXiv.2510.15669},
}
```
