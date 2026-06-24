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
# - Retained the original, unmodified KLD_laplace, vae_masks, and 
#   optimal_permute functions, as well as the LaplaceLoss class.
# - Implemented significant additions for free energy computations 
#   required by the EM procedure.
# - Added auxiliary functions for accuracy computation.
# - Updated the script to support the ECML PKDD 2026 contribution 
#   "Disentanglement in a Multi-Stream VAE".
# Original repository template: https://github.com
# ----------------------------------------------------------------------


import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.utils.data import TensorDataset
from torch.distributions import Laplace
import math


def mix_mnist_data_return_individual_sources(dataset_sources, 
                                             num_samples, 
                                             pi_gen_values, 
                                             scale):
    output_data = []
    output_sources = []
    output_individual_images = []

    for _ in range(num_samples):
        bit_vector = []
        individual_images = []

        for i, pi_gen in enumerate(pi_gen_values):
            s = torch.Tensor(np.random.choice([0, 1], size=1, p=[1-pi_gen, pi_gen]))
            bit_vector.append(s)
            if s.item() == 1:  # activated source
                img = dataset_sources[i][np.random.choice(len(dataset_sources[i]))]
            else:
                img = torch.zeros_like(dataset_sources[i][0])
            individual_images.append(img)

        mixed_mean = sum(
            s * img for s, img in zip(bit_vector, individual_images)
        )

        m = Laplace(mixed_mean, scale).sample()

        output_data.append(m)
        output_sources.append(torch.stack(bit_vector))
        output_individual_images.append(torch.stack(individual_images))  

    output_data = torch.stack(output_data)
    output_sources = torch.stack(output_sources)
    output_individual_images = torch.stack(output_individual_images) 

    final_dataset = TensorDataset(output_data, output_sources, output_individual_images)
    
    return final_dataset

def mix_mnist_data_fixed_sources(dataset_sources, num_samples, K, H, scale, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        # If using CUDA: torch.cuda.manual_seed_all(seed)

    output_data = []
    output_sources = []
    output_individual_images = []

    for _ in range(num_samples):
        bit_vector = torch.zeros(H, dtype=torch.float32)
        individual_images = []

        # Randomly choose K distinct source indices
        active_indices = np.random.choice(H, size=K, replace=False)
        bit_vector[active_indices] = 1

        # For each source, pick image if active; else zeros
        for i in range(H):
            if bit_vector[i] == 1:
                img = dataset_sources[i][np.random.choice(len(dataset_sources[i]))]
            else:
                img = torch.zeros_like(dataset_sources[i][0])
            individual_images.append(img)

        # Mix images (sum weighted by bit vector)
        mixed_mean = sum(bit_vector[i] * individual_images[i] for i in range(H))

        m = Laplace(mixed_mean, scale).sample()

        # output_data.append(m)
        output_data.append(m)
        output_sources.append(bit_vector)
        output_individual_images.append(torch.stack(individual_images))

    output_data = torch.stack(output_data)
    output_sources = torch.stack(output_sources)
    output_individual_images = torch.stack(output_individual_images)
    final_dataset = TensorDataset(output_data, output_sources, output_individual_images)
    return final_dataset

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
    temp = scale+t1+t2+t3
    KLD = -1.0/(2*scale)*torch.sum(1+t1+t2+t3)
    return KLD

def vae_masks(mu_s, x):
    mu_sm = mu_s * (x[:, None, ...] / mu_s.sum(1).unsqueeze(1))
    mu_sm[torch.isnan(mu_sm)] = 0
    return mu_sm

def vae_masks_audio(s_vae, x_vae, x):  
    mu_sm = s_vae * (x / x_vae) # exp(1/2x)
    mu_sm[np.isnan(mu_sm)] = 0
    return mu_sm

# Optimal permutation based on MSE, with the Hungarian Algorithm (Assignment Problem)
def optimal_permute(y,x):
    n = x.size(0)
    nx = x.size(1)
    ny = y.size(1)
    z = torch.zeros_like(x)
    for i in range(n):
        cost = torch.zeros(ny,nx)
        for j in range(ny):
            cost[j] = (y[i,j].unsqueeze(0) - x[i,:]).pow(2).sum(-1).sum(-1)

        row_ind, col_ind = linear_sum_assignment(cost.detach().numpy().T)
        z[i] = y[i,col_ind]
    return z

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

# Inner energy a-K-on fixed prior
def inner_energy_mnist_fixed_K(x, scale, s_list, mu_theta_reshaped, z_reshaped, D, H, K, n_sources):

    const1 = H*torch.tensor((np.sqrt(2*np.pi))**n_sources, dtype=torch.float32)
    const2 = D*torch.tensor(2*scale, dtype=torch.float32)
    term1 = -torch.log(const1) - torch.log(const2) 

    mu_sum = 0
    z_norm_sum = 0
    
    for i in range(n_sources):
        s = s_list[:, i]
        mu_theta = mu_theta_reshaped[i]
        mu_sum += s*mu_theta

        z = z_reshaped[i]
        z_norm_sum += (torch.norm(z, dim=-1) ** 2)

    term2_sum = -torch.norm((x - mu_sum), p=1, dim=-2) / scale 
    term3_sum = -z_norm_sum / 2

    prior_term = (math.factorial(K) * math.factorial(n_sources-K)) / math.factorial(n_sources)
    term4 = torch.log(torch.tensor(prior_term))
        
    E_theta_all = term1 + term2_sum + term3_sum[..., None] + term4  

    up_bound = 0.0
    shift = up_bound - E_theta_all.max(dim=-1, keepdim=True)[0] 
    
    return E_theta_all + shift, mu_sum, term4

# inner energy Bernoulli prior
def inner_energy_mnist_k_sources(x, scale, pi_list, s_list, mu_theta_reshaped, z_reshaped, D, H, M, n_sources):

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

# Audio inner energy computation
def inner_energy_audio(x, scale, pi_1, pi_2, s_list, mu_theta_1, mu_theta_2, z_1, z_2, D, H): 

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

# MNIST inner energy computation - not fixed K
def free_energy_first_term_and_pies_mnist_k_sources(x, b, N, M, s_list, pi_list, mu_theta_list, z_list, n_sources):

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
    E_theta_shifted, mu_sum, s_sum = inner_energy_mnist_k_sources(x, b, pi_list, s_list, transformed_mu_theta, transformed_z_list, D, H, M, n_sources)  
    
    exp_E_theta_shifted = torch.exp(E_theta_shifted) 
    E_theta_sum_s = torch.sum(exp_E_theta_shifted, dim=-1, keepdim=True) 

    # Calculate q(s|x) 
    q_s_z_x = exp_E_theta_shifted/E_theta_sum_s # q(s|z,x)
    q_s_x = q_s_z_x.sum(dim=tuple(range(1, q_s_z_x.ndimension() - 1)))/(M**n_sources)
    q_s_x = q_s_x.detach() # q(s|x) probabilities for each binary case, kept it individual for every input x
    
    log_q_dist = E_theta_shifted - torch.log(E_theta_sum_s) 
    
    log_noise_dist = -D*torch.log(torch.tensor(2*b)) - torch.norm((x - mu_sum), p=1, dim=-2)/b

    # Check for NaN in log_noise distribution 
    if torch.isnan(log_noise_dist).any() or torch.isinf(log_noise_dist).any():
        print(f"NaNs or infs detected in log_noise_dist")
        print(f"log_noise_dist: {log_noise_dist}")

    log_prior_s_dist = s_sum

    # Check for NaN in log_prior distribution -> when one of the pi-parameters goes to zero
    if torch.isnan(log_prior_s_dist).any() or torch.isinf(log_prior_s_dist).any():
        print(f"NaNs or infs detected in log_prior_s_dist")
        print(f"log_prior_s_dist: {log_prior_s_dist}")
    
    # Check for NaN in log_posterior distribution
    if torch.isnan(log_q_dist).any() or torch.isinf(log_q_dist).any():
        print(f"NaNs or infs detected in log_q_dist")
        print(f"log_q_dist: {log_q_dist}")

    numerator = (log_noise_dist + log_prior_s_dist - log_q_dist)*exp_E_theta_shifted
    numerator_sum = torch.sum(numerator, dim=-1) 
    denominator = torch.sum(exp_E_theta_shifted, dim=-1) 

    ELL = torch.sum(numerator_sum/denominator)/(N*(M)**n_sources) 

    exp_E_theta_shifted.detach()

    # pi update 
    pi_new_list = []
    for i in range(n_sources):
        s_i = s_list[..., i]
        numerator_pi_i = torch.sum(exp_E_theta_shifted*s_i, dim=-1)
        pi_i_new = torch.sum(numerator_pi_i/denominator)/(N*(M)**n_sources)
        pi_new_list.append(pi_i_new)

    # b (scale) update
    numerator_b = torch.sum(exp_E_theta_shifted*(torch.norm((x - (mu_sum)), p=1, dim=-2)), dim=-1)
    b_new = torch.sum(numerator_b/denominator)/(N*D*(M)**n_sources)

    return ELL, pi_new_list, b_new, q_s_x 

# MNIST inner energy with fixed K sources per image
def free_energy_first_term_and_pies_mnist_fixed_K(x, b, N, M, K, s_list, mu_theta_list, z_list, n_sources):

    D = x.size(1)
    transformed_mu_theta = []
    transformed_z_list = []

    H = z_list[0].size(2)

    N = x.size(0)
    
    for i in range(n_sources):
        shape = [N] + [1] * i + [M] + [1] * (n_sources - i - 1) + [D, 1]
        mu_theta = mu_theta_list[i]
        mu_theta = mu_theta.view(*shape)
        transformed_mu_theta.append(mu_theta)

        shape_z = [N] + [1] * i + [M] + [1] * (n_sources - i - 1) + [H]
        z_i = z_list[i]
        z_i = z_i.view(*shape_z)
        transformed_z_list.append(z_i)

    shape_x = [N] + [1] * n_sources + [D, 1]
    x = x.view(*shape_x)

    E_theta_shifted, mu_sum, s_sum = inner_energy_mnist_fixed_K(x, 
                                                                b, 
                                                                s_list, 
                                                                transformed_mu_theta, 
                                                                transformed_z_list, 
                                                                D, 
                                                                H, 
                                                                K, 
                                                                n_sources)  
    
   
    exp_E_theta_shifted = torch.exp(E_theta_shifted) 
    E_theta_sum_s = torch.sum(exp_E_theta_shifted, dim=-1, keepdim=True) 
    
    # Calculate q(s|x) 
    q_s_z_x = exp_E_theta_shifted/E_theta_sum_s 
    q_s_x = q_s_z_x.sum(dim=tuple(range(1, q_s_z_x.ndimension() - 1)))/(M**n_sources)
    q_s_x = q_s_x.detach() 

    log_q_dist = E_theta_shifted - torch.log(E_theta_sum_s) 
    
    log_noise_dist = -D*torch.log(torch.tensor(2*b)) - torch.norm((x - mu_sum), p=1, dim=-2)/b

    # Check for NaN in log_noise distribution 
    if torch.isnan(log_noise_dist).any() or torch.isinf(log_noise_dist).any():
        print(f"NaNs or infs detected in log_noise_dist")
        print(f"log_noise_dist: {log_noise_dist}")

    log_prior_s_dist = s_sum

    # Check for NaN in log_prior distribution -> when one of the pi-parameters goes to zero
    if torch.isnan(log_prior_s_dist).any() or torch.isinf(log_prior_s_dist).any():
        print(f"NaNs or infs detected in log_prior_s_dist")
        print(f"log_prior_s_dist: {log_prior_s_dist}")
    
    # Check for NaN in log_posterior distribution
    if torch.isnan(log_q_dist).any() or torch.isinf(log_q_dist).any():
        print(f"NaNs or infs detected in log_q_dist")
        print(f"log_q_dist: {log_q_dist}")

    numerator = (log_noise_dist + log_prior_s_dist - log_q_dist)*exp_E_theta_shifted 
    numerator_sum = torch.sum(numerator, dim=-1) 
    denominator = torch.sum(exp_E_theta_shifted, dim=-1) 

    ELL = torch.sum(numerator_sum/denominator)/(N*(M)**n_sources) 

    exp_E_theta_shifted.detach()

    numerator_b = torch.sum(exp_E_theta_shifted*(torch.norm((x - (mu_sum)), p=1, dim=-2)), dim=-1)
    b_new = torch.sum(numerator_b/denominator)/(N*D*(M)**n_sources)

    return ELL, b_new, q_s_x 

# audio inner energy computation
def free_energy_first_term_and_pies_audio(x, b, N, M, s_list, pi_list, mu_theta_list, z_list):
    
    D = x.size(1) 

    mu_theta_1 = mu_theta_list[0] 
    mu_theta_2 = mu_theta_list[1] 
    
    mu_theta_1 = mu_theta_1[:, :, None, :, None] 
    mu_theta_2 = mu_theta_2[:, None, :, :, None]

    pi_1 = pi_list[0] 
    pi_2 = pi_list[1] 

    z_1 = z_list[0] 
    z_2 = z_list[1] 

    H = z_1.size(2)

    z_1 = z_1[:, :, None, :] 
    z_2 = z_2[:, None, :, :]  

    x = x[:, None, None, :, None] 

    s_1 = s_list[:, 0]
    s_2 = s_list[:, 1] 
    
    E_theta_shifted = inner_energy_audio(x, b, 
                                         pi_1, pi_2, 
                                         s_list, 
                                         mu_theta_1, mu_theta_2, 
                                         z_1, z_2, 
                                         D, H) 
    exp_E_theta_shifted = torch.exp(E_theta_shifted) 
    E_theta_sum_s = torch.sum(exp_E_theta_shifted, dim=3, keepdim=True) 

    # Calculate q(s|x) 
    q_s_z_x = exp_E_theta_shifted/E_theta_sum_s # q(s|z,x)
    q_s_x = q_s_z_x.sum(dim=(1, 2))/(M*M) # q(s|x) probabilities for each binary case, kept it individual for every input x

    log_q_dist = E_theta_shifted - torch.log(E_theta_sum_s) 

    log_noise_dist = -D*torch.log(torch.tensor(2*b)) - torch.norm((x - (mu_theta_1*s_1 + mu_theta_2*s_2)), p=1, dim=3)/b

    # Debugging output
    if torch.isnan(log_noise_dist).any() or torch.isinf(log_noise_dist).any():
        print(f"NaNs or infs detected in log_noise_dist")
        print(f"log_noise_dist: {log_noise_dist}")

    log_prior_s_dist = (s_1 * torch.log(pi_list[0]) + (1 - s_1) * torch.log(1 - pi_list[0])) + s_2 * torch.log(pi_list[1]) + (1 - s_2) * torch.log(1 - pi_list[1]) # (1, S)

    # Debugging output
    if torch.isnan(log_prior_s_dist).any() or torch.isinf(log_prior_s_dist).any():
        print(f"NaNs or infs detected in log_prior_s_dist")
        print(f"log_prior_s_dist: {log_prior_s_dist}")
    
    # Debugging output
    if torch.isnan(log_q_dist).any() or torch.isinf(log_q_dist).any():
        print(f"NaNs or infs detected in log_q_dist")
        print(f"log_q_dist: {log_q_dist}")

    numerator = (log_noise_dist + log_prior_s_dist - log_q_dist)*exp_E_theta_shifted 
    numerator_sum = torch.sum(numerator, dim=3) 
    denominator = torch.sum(exp_E_theta_shifted, dim=3) 

    ELL = torch.sum(numerator_sum/denominator)/(N*M*M) 

    exp_E_theta_shifted.detach()

    # pi update 
    numerator_pi_1 = torch.sum(exp_E_theta_shifted*s_1, dim=3)
    numerator_pi_2 = torch.sum(exp_E_theta_shifted*s_2, dim=3)
    pi_1_new = torch.sum(numerator_pi_1/denominator)/(N*M*M)
    pi_2_new = torch.sum(numerator_pi_2/denominator)/(N*M*M)

    # b (scale) update
    numerator_b = torch.sum(exp_E_theta_shifted*(torch.norm((x - (mu_theta_1*s_1 + mu_theta_2*s_2)), p=1, dim=3)), dim=3)
    b_new = torch.sum(numerator_b/denominator)/(N*M*M*D)

    return ELL, pi_1_new, pi_2_new, b_new, q_s_x

# Accuracy of prediction on test set (q_s_x) based on ground truth
def accuracy_score(target, prediction):
    correct = (target == prediction).float()  
    accuracy = correct.sum() / len(correct)  
    return accuracy.item() * 100 

def get_case_numbers_H(labels, q_s_x, H):
    multiplier = 2**torch.arange(H).float()
    case_numbers = torch.mv(labels, multiplier).int()
   
    _, max_indices = torch.max(q_s_x.detach().cpu(), dim=1)

    return case_numbers, max_indices 

def get_case_numbers(labels, q_s_x):
    _, max_indices = torch.max(q_s_x.detach().cpu(), dim=1)
    
    multiplier = torch.tensor([2., 1.])
    case_numbers = torch.mv(labels, multiplier).round().int()

    return case_numbers, max_indices 
