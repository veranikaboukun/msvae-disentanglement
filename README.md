# Disentanglement of Sources in a Multi-Stream Variational Autoencoder

## Overview 
The [pretraining](./pretraining), [training](./training) and [evaluation](./evaluation) directories and corresponding sub-directories contain implementations of Multi-Stream Variational Autoencoder (MS-VAE) to be presented at ECML PKDD 2026.

The MNIST datasets which were used in Section 3.2 for MNIST digit source separation can be found here. The raw audio files from the [database](https://zenodo.org/records/17342367) used in Section 3.3 cannot be provided due to data protection regulations, but the source code (for pretraining, training and evaluation) can be found in the repository. 

Please follow the [Setup](#setup) instructions described below to run the experiments. 
The code has only been tested on Linux systems.  

## Setup
We recommend [Anaconda](https://www.anaconda.com/) to manage the installation, and to create a new environment for hosting the installed packages.

## Installation and Environment Setup

Due to conflicting dependency trees and differing Python version requirements among the benchmarked models, we provide separate environment files. We highly recommend using **Conda** to manage these isolated environments.

### 1. Main Environment (MS-VAE and VAEM-BSS)
This is the primary environment used for (pre-)training and evaluation of Multi-Stream VAE model. 
The VAE(M)-BSS baseline can be evaluated on it as well.

```bash
conda create -n ms_vae python=3.8.10 -y
conda activate ms_vae
pip install -r requirements.txt
```

### 2. BASIS Evaluation Environment
The BASIS model relies on an older ecosystem and requires a specific, legacy Python version to execute properly.

```bash
conda create -n basis_env python=3.6.13 -y
conda activate basis_env
pip install -r requirements_BASIS.txt
```

### 3. LASS Evaluation Environment
The LASS baseline requires its own set of dependencies to properly import the transformer models and `diba` interfaces.
```bash
conda create -n lass_env python=3.8.20 -y
conda activate lass_env
pip install -r requirements_LASS.txt
```

## Pretrained Models & Requirements

We provide the pre-trained checkpoints for our MS-VAE 10% and MS-VAE 100% models. 
**(To be updated later with a Zenodo link to the datasets and checkpoints)**

To replicate the results in Table 1, download the following pretrained weights and external files from the original repositories:

### 1. VAE-BSS Checkpoints
* **Files:** `model_vae_K2.pt`, `model_vae_K3.pt`, and `model_vae_K4.pt`
* **Source:** Download from the [VAE-BSS Repository](https://github.com/jundsp/VAE-BSS/tree/main/saves/pretrained).

### 2. BASIS NCSN Checkpoints & Architecture
* **Weights:** Download `mnist/checkpoint.pth` from the [Basis Separation Repository](https://github.com/jthickstun/basis-separation/tree/master).
* **Architecture:** Download `cond_refinenet_dilated.py` from [ncsn/models/](https://github.com/jthickstun/basis-separation/blob/master/ncsn/models/cond_refinenet_dilated.py).

### 3. LASS Evaluation
Due to licensing restrictions on the external [LASS repository](https://github.com/gladia-research-group/latent-autoregressive-source-separation/tree/main), we cannot provide a fully integrated evaluation script for it. 
To evaluate LASS on MNIST digit separation, please adapt their official script by following these instructions:

#### Environment Setup
Use the file `separate.py` from the `lass_mnist/lass/` directory. Ensure you have installed all necessary dependencies to import the transformer models and `diba` interfaces.

#### `main()` Function Modifications
Modify the `main()` function in `separate.py` with the following configuration and initialization steps:

1. **Model Initialization:** Instantiate the VQ-VAE and GPT-2 Transformer models.
```python
# Instantiate VQ-VAE
model = VectorQuantizedVAE(input_dim=1, dim=128, K=256).to(device)

# Instantiate Transformer
transformer = GPT2LMHeadModel(GPT2Config(
    vocab_size=256, n_positions=49, n_embd=128, 
    n_layer=3, n_head=2, use_cache=False, 
    bos_token_id=0, eos_token_id=511
)).to(device)
```

2. **Load Weights:** Download the VQ-VAE model weights [here](https://drive.google.com/file/d/1oayY1FEUrTwQJMr78mP1t6r8AggjzAso/view) and load the checkpoints.
```python
# Load VQ-VAE weights (checkpoints/vqvae/256-sigmoid-big.pt)
model.load_state_dict(torch.load(args.vq_path, map_location=cfg.device))

# Load transformer weights (checkpoints/unconditioned/256-sigmoid-big.pt)
transformer.load_state_dict(torch.load(args.auto_path, map_location=cfg.device))

# Load sums (checkpoints/unconditioned/256-sigmoid-big.pt)
sums = torch.load(args.sum_path, map_location=cfg.device)
```

3. **Data Loading:** Load the test set `mnist_test_set_K_2_noiseless`.pt. You can replicate the lines 263–305 of our `evaluate_vae_bss.py` script.

#### Evaluation Loop Modifications
Update the evaluation loop to handle active MNIST sources and calculate metrics:

1. **Loop Definition:** Update the dataloader loop structure:
```python
for i, (data, labels, individual_sources) in enumerate(mnist_dataloader_test):
```

2. **Source Masking:** Define and filter for exactly 2 active sources inside the loop (similar to VAEM-BSS):
```python
sources_mask = labels.squeeze(-1) 
num_active_sources = sources_mask.sum(dim=1)   
selected_indices_k2 = (num_active_sources == 2)

selected_sources_mask_k2 = sources_mask[selected_indices_k2]
selected_individual_sources_k2 = individual_sources[selected_indices_k2] 

activated_images_selected_k2 = []
for i in range(selected_individual_sources_k2.shape[0]):
    mask = selected_sources_mask_k2[i] > 0
    images = selected_individual_sources_k2[i][mask]
    activated_images_selected_k2.append(images)

activated_images_selected_k2_batch = torch.stack(activated_images_selected_k2, dim=0)
gt1 = activated_images_selected_k2_batch[:, 0, ...].to(device)
gt2 = activated_images_selected_k2_batch[:, 1, ...].to(device)
```

3. **Sample Generation:** Execute the remainder of the native script inside `generate_samples()` using `latent_length=7`.

4. **Metrics Evaluation:** Combine the predictions to prepare them for the `compute_permutation_invariant_metrics()` function to extract the final PSNR/SSIM values:
```python
preds_combined = torch.stack([gen1, gen2], dim=1)
activated_preds_selected_k2 = []

for i in range(preds_combined.shape[0]):
    activated_preds_selected_k2.append(preds_combined[i])
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
