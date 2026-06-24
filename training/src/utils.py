# Copyright (C) 2021 Julian Neri, Roland Badeau, Philippe Depalle
# Copyright (C) 2026 Veranika Boukun (veranika.boukun@uni-oldenburg.de), Jörg Lücke
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://gnu.org>.
#
# ----------------------------------------------------------------------
# Modifications made by Veranika Boukun (2026):
# - Added new mixing functions tailored to the proposed generative model.
# - Modified the KLD_gauss function (including the addition by N).
# - Retained the original, unmodified KLD_laplace function as well as 
#   the LaplaceLoss class.
# - Implemented significant additions for free energy computations 
#   required by the EM procedure, as well as entropy calculations for 
#   further analysis.
# - Added auxiliary functions for accuracy computation, H5 loading, 
#   and CSV processing.
# - Updated the script to support the ECML PKDD 2026 contribution 
#   "Disentanglement in a Multi-Stream VAE".
# Original repository template: https://github.com
# ----------------------------------------------------------------------


import h5py
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.distributions import Laplace
from collections import Counter
from torch.distributions import Normal, Laplace
import csv

def mix_mnist_data_bernoulli_arbitrary(dataset_sources, 
                                       num_samples, 
                                       pi_gen_values, 
                                       scale):
    output_data = []
    output_sources = []

    # Generate data of "num_samples" size
    for _ in range(num_samples):
        # Generate a bit vector of size `n_sources`
        bit_vector = []
        for pi_gen in pi_gen_values:
            s = torch.Tensor(np.random.choice([0, 1], size=1, p=[1-pi_gen, pi_gen]))
            bit_vector.append(s)
        
        # Calculate mixed means for each sample, where each source contributes based on its bit vector
        mixed_mean = sum(
            s * dataset_sources[i][np.random.choice(len(dataset_sources[i]))]
            for i, s in enumerate(bit_vector)
        )

        # Get a sample from Laplace distribution with mixed mean
        m = Laplace(mixed_mean, scale).sample()

        output_data.append(m)
        output_sources.append(torch.stack(bit_vector))

    # Convert lists to tensors
    output_data = torch.stack(output_data)
    output_sources = torch.stack(output_sources) # binary vectors
    
    # Create your final dataset as a TensorDataset
    final_dataset = TensorDataset(output_data, output_sources)
    
    return final_dataset

def calculate_class_percentage(output_sources):

    bit_vectors_tuple = [tuple(x.numpy()) for x in output_sources]
    counter = Counter(bit_vectors_tuple)
    total = sum(counter.values())
    percentages = {k: (v / total) * 100 for k, v in counter.items()}

    return percentages

def calculate_class_percentage_audio(output_sources):

    bit_vectors_tuple = [tuple(x.numpy().flatten()) for x in output_sources]
    counter = Counter(bit_vectors_tuple)
    total = sum(counter.values())
    percentages = {k: (v / total) * 100 for k, v in counter.items()}

    return percentages

def load_h5_data(data_with_labels, dimx, dimz):

    with h5py.File(data_with_labels, "r") as f:
        data = torch.tensor(np.array(f["data"]), dtype=torch.get_default_dtype())
        data_last = torch.tensor(np.array(f["data_last_batch"]), dtype=torch.get_default_dtype())
        mu_z = torch.tensor(np.array(f["mu_z"]), dtype=torch.get_default_dtype())
        mu_z_last = torch.tensor(np.array(f["mu_z_last_batch"]), dtype=torch.get_default_dtype())
        logvar_z = torch.tensor(np.array(f["logvar_z"]), dtype=torch.get_default_dtype())
        logvar_z_last = torch.tensor(np.array(f["logvar_z_last_batch"]), dtype=torch.get_default_dtype())
    
        data = data.view(-1, dimx)
        data_last = data_last.view(-1, dimx)
        combined_data = torch.cat((data, data_last), 0)

        mu_z = mu_z.view(-1, dimz)
        mu_z_last = mu_z_last.view(-1, dimz)
        combined_mu_z = torch.cat((mu_z, mu_z_last), 0)

        logvar_z = logvar_z.view(-1, dimz)
        logvar_z_last = logvar_z_last.view(-1, dimz)
        combined_logvar_z = torch.cat((logvar_z, logvar_z_last), 0)

    return combined_data, combined_mu_z, combined_logvar_z

def load_h5_data_no_last_batch(data_with_labels, dimx, dimz):

    with h5py.File(data_with_labels, "r") as f:
        data = torch.tensor(np.array(f["data"]), dtype=torch.get_default_dtype())
        mu_z = torch.tensor(np.array(f["mu_z"]), dtype=torch.get_default_dtype())
        logvar_z = torch.tensor(np.array(f["logvar_z"]), dtype=torch.get_default_dtype())
        data = data.view(-1, dimx)
        mu_z = mu_z.view(-1, dimz)
        logvar_z = logvar_z.view(-1, dimz)

    return data, mu_z, logvar_z

def compute_pi_from_labels(csv_file):
    """
    Reads a CSV with columns:
      FrameIndex, Instrument1, Instrument2, Instrument3
    Computes:
      pi_per_instrument = N_active_instrument / Total_frames
      pi_any_active     = N_frames_with_any_instrument_active / Total_frames
    Returns them in a dictionary.
    """
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)

        # Read the header row
        header = next(reader, None)
        if header is None:
            print(f"Empty CSV: {csv_file}")
            return []

        # The first column is assumed to be "FrameIndex", so the rest are instruments
        instrument_names = header[1:]
        num_instruments = len(instrument_names)

        # Prepare counters
        sums = [0] * num_instruments  # sum of activity for each instrument
        sum_any = 0
        total_rows = 0

        # Read each data row
        for row in reader:
            # Ensure the row has at least 1 + num_instruments columns
            if len(row) < 1 + num_instruments:
                continue

            # Convert each instrument column to int
            activity_flags = [int(x) for x in row[1:1+num_instruments]]

            # Update sums
            for i, val in enumerate(activity_flags):
                sums[i] += val

            # Check if any instrument is active
            if sum(activity_flags) > 0:
                sum_any += 1

            total_rows += 1

        if total_rows == 0:
            print(f"No valid rows found in {csv_file}")
            return []

        # Compute pi_instrument for each instrument
        pi_values = []
        for i in range(num_instruments):
            pi_instrument = sums[i] / total_rows
            pi_values.append(pi_instrument)

        # Compute pi_any_active
        pi_any = sum_any / total_rows
        pi_values.append(pi_any)

        return pi_values

# KL(q(x)||p(x)) where p(x) is Gaussian, q(x) is Gaussian
def KLD_gauss(mu,logvar):
    N = mu.size(0)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD/N

# KL(q(x)||p(x)) where p(x) is Laplace, q(x) is Gaussian
def KLD_laplace(mu,logvar,scale=1.0):
    v = logvar.exp()
    y = mu/torch.sqrt(2*v)
    y2 = y.pow(2)
    t1 = -2*torch.exp(-y2)*torch.sqrt(2*v/np.pi)
    t2 = -2*mu*torch.erf(y)
    t3 = scale*torch.log(np.pi*v/(2.0*scale*scale))
    KLD = -1.0/(2*scale)*torch.sum(1+t1+t2+t3)
    return KLD

class LaplaceLoss(torch.nn.Module):
    def __init__(self, variance=None,reduction='sum'):
        super(LaplaceLoss, self).__init__()
        if variance is None:
            variance = torch.tensor(1.0)
        self.register_buffer("log2",torch.tensor(2.0).log())
        self.register_buffer("scale",torch.sqrt(0.5*variance))
        self.logscale = self.scale.log()

    def forward(self, estimate, target):
        return torch.sum((target-estimate).abs() / self.scale)

# MNIST - 2 sources
def inner_energy_mnist(x, scale, pi_1, pi_2, s_list, mu_theta_1, mu_theta_2, z_1, z_2, D, H): 

    const1 = torch.tensor(2*np.pi, dtype=torch.float32) 
    const2 = torch.tensor(2*scale, dtype=torch.float32) 

    s1 = s_list[:, 0]
    s2 = s_list[:, 1] #TODO
   
    term1 = -torch.log(const1) - torch.log(const2)
    term2 = -torch.norm((x - (mu_theta_1*s1 + mu_theta_2*s2)), p=1, dim=3)/ scale 

    term3 = -(torch.norm(z_1, dim=3)**2 + torch.norm(z_2, dim=3)**2) / 2 
    term4 = s1 * torch.log(pi_1) + (1 - s1) * torch.log(1 - pi_1) 
    term5 = s2 * torch.log(pi_2) + (1 - s2) * torch.log(1 - pi_2) 
    E_theta_all = term1 + term2 + term3[..., None] + term4 + term5 

    up_bound = 0.0
    shift = up_bound - E_theta_all.max(dim=3, keepdim=True)[0]
    
    return E_theta_all + shift

def inner_energy_k_sources(x, scale, pi_list, s_list, mu_theta_reshaped, z_reshaped, D, H, M, n_sources):

    const1 = H*torch.tensor((np.sqrt(2*np.pi))**n_sources, dtype=torch.float32)
    const2 = D*torch.tensor(2*scale, dtype=torch.float32)
    term1 = -torch.log(const1) - torch.log(const2) 

    mu_sum = 0
    z_norm_sum = 0
    term4_sum = 0
    for i in range(n_sources):
        # term 2: mu term
        s = s_list[:, i]
        mu_theta = mu_theta_reshaped[i]
        mu_sum += s*mu_theta

        # term 3: z term
        z = z_reshaped[i]
        z_norm_sum += (torch.norm(z, dim=-1) ** 2) 

        # term 4: s term
        s = s_list[:, i]
        pi = torch.tensor(pi_list[i])
        term4_sum += s * torch.log(pi) + (1 - s) * torch.log(1 - pi)

    term2_sum = -torch.norm((x - mu_sum), p=1, dim=-2) / scale 
    term3_sum = -z_norm_sum / 2 
        
    E_theta_all = term1 + term2_sum + term3_sum[..., None] + term4_sum  

    up_bound = 0.0
    shift = up_bound - E_theta_all.max(dim=-1, keepdim=True)[0]  
    
    return E_theta_all + shift, mu_sum, term4_sum

def free_energy_first_term_and_pies_k_sources(x, b, N, M, s_list, pi_list, mu_theta_list, z_list, n_sources):

    D = x.size(1) # dimx = 784
    transformed_mu_theta = []
    transformed_z_list = []

    H = z_list[0].size(2)

    N = x.size(0)
    
    # automatically creating correct 'shapes' of mu's and z's (dependend on the source index)
    for i in range(n_sources):
        shape = [N] + [1] * i + [M] + [1] * (n_sources - i - 1) + [D, 1]
        mu_theta = mu_theta_list[i]
        mu_theta = mu_theta.view(*shape)
        transformed_mu_theta.append(mu_theta)

        shape_z = [N] + [1] * i + [M] + [1] * (n_sources - i - 1) + [H]
        z_i = z_list[i]
        z_i = z_i.view(*shape_z)
        transformed_z_list.append(z_i)

    # x automatically reshaped to correct number of M's: (N, M_1,... M_i, D, H) - independend of the source index
    shape_x = [N] + [1] * n_sources + [D, 1]
    x = x.view(*shape_x)

    # works for M=1 if n_sources=10
    E_theta_shifted, mu_sum, s_sum = inner_energy_k_sources(x, b, pi_list, s_list, transformed_mu_theta, transformed_z_list, D, H, M, n_sources)  
    
    # TODO: right now comment out everything below:
    exp_E_theta_shifted = torch.exp(E_theta_shifted) # (N/B, M_1, ..., M_n, S)
    E_theta_sum_s = torch.sum(exp_E_theta_shifted, dim=-1, keepdim=True) # (N/B, M_1,... M_n, S=1)
    # Calculate q(s|x) 
    q_s_z_x = exp_E_theta_shifted/E_theta_sum_s # q(s|z,x)
    q_s_x = q_s_z_x.sum(dim=tuple(range(1, q_s_z_x.ndimension() - 1)))/(M**n_sources)
    q_s_x = q_s_x.detach() # q(s|x) probabilities for each binary case, kept it individual for every input x, shape: (B/N, S)
    
    log_q_dist = E_theta_shifted - torch.log(E_theta_sum_s) # (N/B, M, M, S)
    
    log_noise_dist = -D*torch.log(torch.tensor(2*b)) - torch.norm((x - mu_sum), p=1, dim=-2)/b

    # Check for NaN in log_noise distribution 
    if torch.isnan(log_noise_dist).any() or torch.isinf(log_noise_dist).any():
        print(f"NaNs or infs detected in log_noise_dist")
        print(f"log_noise_dist: {log_noise_dist}")

    log_prior_s_dist = s_sum # (1, S)

    # Check for NaN in log_prior distribution -> when one of the pi-parameters goes to zero
    if torch.isnan(log_prior_s_dist).any() or torch.isinf(log_prior_s_dist).any():
        print(f"NaNs or infs detected in log_prior_s_dist")
        print(f"log_prior_s_dist: {log_prior_s_dist}")
    
    # Check for NaN in log_posterior distribution
    if torch.isnan(log_q_dist).any() or torch.isinf(log_q_dist).any():
        print(f"NaNs or infs detected in log_q_dist")
        print(f"log_q_dist: {log_q_dist}")

    numerator = (log_noise_dist + log_prior_s_dist - log_q_dist)*exp_E_theta_shifted # (N/B, M_1, ..., M_n, S)
    numerator_sum = torch.sum(numerator, dim=-1) # (N/B, M_1, ..., M_n)
    denominator = torch.sum(exp_E_theta_shifted, dim=-1) # (N/B, M_1, ..., M_n)

    ELL = torch.sum(numerator_sum/denominator)/(N*(M)**n_sources) # const

    exp_E_theta_shifted.detach()

    # # pi update 
    pi_new_list = []
    for i in range(n_sources):
        s_i = s_list[..., i]
        numerator_pi_i = torch.sum(exp_E_theta_shifted*s_i, dim=-1)
        pi_i_new = torch.sum(numerator_pi_i/denominator)/(N*(M)**n_sources)
        pi_new_list.append(pi_i_new)
    
    # # fixed pi
    # # pi_1_new = torch.tensor(0.3)
    # # pi_2_new = torch.tensor(0.2) 

    # b (scale) update
    numerator_b = torch.sum(exp_E_theta_shifted*(torch.norm((x - (mu_sum)), p=1, dim=-2)), dim=-1)
    b_new = torch.sum(numerator_b/denominator)/(N*D*(M)**n_sources)
    
    # fixed b
    # b_new = torch.tensor(0.1)

    return ELL, pi_new_list, b_new, q_s_x 
    # return None

# Damm et al. 2023, AISTATS: The ELBO of Variational Autoencoders Converges to a Sum of Entropies
def get_ELBO_entropies_k_sources(x, mu_phi_list, logvar_phi_list, b, s_list, pi_list, zdim, q_s_x, device, n_sources):
    
    p_s = 1
    for s, pi in zip(s_list, pi_list):
        p_s *= (pi ** s) * ((1 - pi) ** (1 - s))

    # Gaussian prior encoder probability 
    p_z = Normal(torch.zeros(zdim).to(device), torch.ones(zdim).to(device))

    # Encoder distributions q(z1|x), q(z2|x)
    H_enc_list = []
    for i in range(n_sources):
        q_zi_x = Normal(mu_phi_list[i].detach(), (0.5 * logvar_phi_list[i].detach()).exp())
        H_enc_i = q_zi_x.entropy().mean(0).sum()
        H_enc_list.append(H_enc_i)

    # Decoder p_x_z
    p_x_z = Laplace(x, b)

    # Calculate entropies
    H_s_prior = calc_entropy(p_s) 
    H_z_prior = p_z.entropy().sum() 

    H_dec = p_x_z.entropy().mean(0).sum()

    H_s_posterior = calc_entropy_dim(q_s_x).mean(0).sum() 

    H_sum = - H_dec - int(n_sources)*H_z_prior + sum(H_enc for H_enc in H_enc_list) - H_s_prior + H_s_posterior

    return H_enc_list, H_dec, H_s_posterior, H_s_prior, H_z_prior, H_sum

def calc_entropy(pmf):
    pmf = pmf / pmf.sum()
    # Calculate entropy
    entropy = -torch.sum(pmf * torch.log(pmf)) 
    return entropy

def calc_entropy_dim(pmf):
    # Normalize the PMF
    pmf = pmf / pmf.sum(dim=1, keepdim=True) # Here dim=1 specifies that we sum along the second dimension

    # Calculate entropy
    entropy = -(pmf * torch.log(pmf+1e-19)) 
    # Here dim=1 specifies that we sum along the second dimension
    return entropy

# Accuracy of prediction on test set (q_s_x) based on ground truth
def accuracy_score(target, prediction):
    correct = (target == prediction).float()  # This will be a tensor of 0s and 1s, where 1 indicates a correct prediction
    accuracy = correct.sum() / len(correct)  # Counting the number of correct predictions and dividing by the total number
    return accuracy.item() * 100 