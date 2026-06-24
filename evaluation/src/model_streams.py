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
# - Restructured and modified the original VAE-BSS pipeline for 
#   MS-VAE architecture and compatibility.
# - Implemented core multi-stream VAE functions and equations, 
#   including an EM procedure not originally required or part of VAE-BSS.
# - Updated the script to support the ECML PKDD 2026 contribution 
#   "Disentanglement in a Multi-Stream VAE".
# Original repository template: https://github.com
# ----------------------------------------------------------------------

import torch
from torch import nn
from .utils import *

class LinearBlock(nn.Module):
    def __init__(self, in_channels,out_channels,activation=True):
        super(LinearBlock, self).__init__()
        if activation is True:
            self.block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                )
        else:
            self.block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                )
    def reset_parameters(self):
        for layer in self.block:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        return self.block(x) 

class MultiStream_VAE(nn.Module):
    def __init__(self, dimx, dimz, n_sources=1, device='cpu', variational=True, freeze_decoder=True, freeze_encoder=False):
        super(MultiStream_VAE, self).__init__()

        self.dimx = dimx
        self.dimz = dimz
        self.n_sources = n_sources
        self.device = device
        self.variational = variational
        self.freeze_decoder = freeze_decoder
        self.freeze_encoder = freeze_encoder

        chans = (700, 600, 500, 400, 300)
        # chans = (350, 300, 250, 200, 150)
        # chans = (175, 150, 125, 100, 75)

        self.out_z = nn.ModuleList([nn.Linear(chans[-1], 2*self.dimz) for _ in range(self.n_sources)])
        self.out_w = nn.ModuleList([nn.Linear(chans[-1], 1) for _ in range(self.n_sources)])

        self.Encoders = nn.ModuleList([
            nn.Sequential(
                LinearBlock(self.dimx, chans[0]),
                LinearBlock(chans[0], chans[1]),
                LinearBlock(chans[1], chans[2]),
                LinearBlock(chans[2], chans[3]),
                LinearBlock(chans[3], chans[4]),
            ) for _ in range(self.n_sources)
        ])

        self.Decoders = nn.ModuleList([
            nn.Sequential(
                LinearBlock(self.dimz, chans[4]),
                LinearBlock(chans[4], chans[3]),
                LinearBlock(chans[3], chans[2]),
                LinearBlock(chans[2], chans[1]),
                LinearBlock(chans[1], chans[0]),
                LinearBlock(chans[0], self.dimx, activation=False),
            ) for _ in range(self.n_sources)
        ])

        # Freeze all the weights in the decoder modules
        if self.freeze_encoder: 
            for encoder in self.Encoders:
                for param in encoder.parameters():
                    param.requires_grad = False

        if self.freeze_decoder:
            for decoder in self.Decoders:
                for param in decoder.parameters():
                    param.requires_grad = False

    def unfreeze_encoder(self):
        for encoder in self.Encoders:
            for param in encoder.parameters():
                param.requires_grad = True

    def unfreeze_decoder(self):
        for decoder in self.Decoders:
            for param in decoder.parameters():
                param.requires_grad = True

    def init_decoder(self, index):
        for block in self.Decoders[index]:
            block.reset_parameters()  # reinitialize parameters
        for param in self.Decoders[index].parameters():
            param.requires_grad = True  # set these parameters to be trainable

    def encode(self, x):
        mu_list = []
        logvar_list = []

        for i in range(self.n_sources):
            d = self.Encoders[i](x)
            dz = self.out_z[i](d)
            mu = dz[:, :self.dimz]
            logvar = dz[:, self.dimz:]
            mu_list.append(mu)
            logvar_list.append(logvar)
        return mu_list, logvar_list
    
    def reparameterize(self, mu, logvar, M):
        std = torch.exp(0.5*logvar)
        eps = torch.randn((std.size(0), M, std.size(1))).to(std.device)
        return mu[:, None, :] + eps*std[:, None, :]

    def decode(self, z):
        recon_separate_list = []
        
        B = z[0].size(0)
        M = z[0].size(1)
        eps = torch.finfo(torch.float32).eps

        for i in range(self.n_sources):
            d = self.Decoders[i](z[i].view(-1, self.dimz))
            recon_separate = torch.sigmoid(d).view(-1, self.dimx) + eps 
            recon_separate_reshaped = recon_separate.view(B, M, self.dimx)
            recon_separate_list.append(recon_separate_reshaped)

        return recon_separate_list 

    def forward(self, x, M):
        mu_phi_list, logvar_phi_list = self.encode(x.view(-1, self.dimx)) 
        if self.variational:
            z_list = [self.reparameterize(mu, logvar, M) for mu, logvar in zip(mu_phi_list, logvar_phi_list)] 
        else:
            z_list = mu_phi_list
        mu_theta_list = self.decode(z_list) 

        return mu_theta_list, mu_phi_list, logvar_phi_list, z_list 

# Multi-stream VAE for audio using simple feed forward archicture 
class MultiStream_VAE_audio(nn.Module):
    def __init__(self, dimx, dimz, n_sources=1, device='cpu', variational=True, freeze_decoder=False):
        super(MultiStream_VAE_audio, self).__init__()

        self.dimx = dimx
        self.dimz = dimz
        self.n_sources = n_sources
        self.device = device
        self.variational = variational
        self.freeze_decoder = freeze_decoder

        chans = (428, 320, 224, 160, 80) 

        self.out_z = nn.ModuleList([nn.Linear(chans[-1], 2*self.dimz) for _ in range(self.n_sources)])
        self.out_w = nn.ModuleList([nn.Linear(chans[-1], 1) for _ in range(self.n_sources)])

        self.Encoders = nn.ModuleList([
            nn.Sequential(
                LinearBlock(self.dimx, chans[0]),
                LinearBlock(chans[0], chans[1]),
                LinearBlock(chans[1], chans[2]),
                LinearBlock(chans[2], chans[3]),
                LinearBlock(chans[3], chans[4]),
            ) for _ in range(self.n_sources)
        ])

        self.Decoders = nn.ModuleList([
            nn.Sequential(
                LinearBlock(self.dimz, chans[4]),
                LinearBlock(chans[4], chans[3]),
                LinearBlock(chans[3], chans[2]),
                LinearBlock(chans[2], chans[1]),
                LinearBlock(chans[1], chans[0]),
                LinearBlock(chans[0], self.dimx, activation=False),
            ) for _ in range(self.n_sources)
        ])

        # Freeze all the weights in the decoder modules
        if self.freeze_decoder:
            for decoder in self.Decoders:
                for param in decoder.parameters():
                    param.requires_grad = False

    def unfreeze_decoder(self):
        for decoder in self.Decoders:
            for param in decoder.parameters():
                param.requires_grad = True

    def encode(self, x):
        mu_list = []
        logvar_list = []

        for i in range(self.n_sources):
            d = self.Encoders[i](x)
            dz = self.out_z[i](d)
            mu = dz[:, :self.dimz]
            logvar = dz[:, self.dimz:]
            mu_list.append(mu)
            logvar_list.append(logvar)
        return mu_list, logvar_list

    def reparameterize(self, mu, logvar, M):
        std = torch.exp(0.5*logvar)
        eps = torch.randn((std.size(0), M, std.size(1))).to(std.device)
        return mu[:, None, :] + eps*std[:, None, :]

    def decode(self, z):
        recon_separate_list = []
        
        B = z[0].size(0)
        M = z[0].size(1)

        for i in range(self.n_sources):
            d = self.Decoders[i](z[i].view(-1, self.dimz))
            recon_separate = torch.sigmoid(d).view(-1, self.dimx) 
            recon_separate_reshaped = recon_separate.view(B, M, self.dimx)
            recon_separate_list.append(recon_separate_reshaped)
        return recon_separate_list 
    
    def forward(self, x, M):
        mu_phi_list, logvar_phi_list = self.encode(x.view(-1, self.dimx)) 
        if self.variational:
            z_list = [self.reparameterize(mu, logvar, M) for mu, logvar in zip(mu_phi_list, logvar_phi_list)]  
        else:
            z_list = mu_phi_list
        mu_theta_list = self.decode(z_list)

        return mu_theta_list, mu_phi_list, logvar_phi_list, z_list 

class LaplaceLoss(nn.Module):
    def __init__(self, variance=None,reduction='sum'):
        super(LaplaceLoss, self).__init__()
        if variance is None:
            variance = torch.tensor(1.0)
        self.register_buffer("log2",torch.tensor(2.0).log())
        self.register_buffer("scale",torch.sqrt(0.5*variance))
        self.logscale = self.scale.log()

    def forward(self, estimate, target):
        return torch.sum((target-estimate).abs() / self.scale) 
    
class Loss_k_sources(nn.Module):
    def __init__(self, sources=2, alpha=None, likelihood='bernoulli', variational=True, prior='gauss', scale=1.0, N=128, M=5):
        super(Loss_k_sources, self).__init__()
        self.variational = variational
        self.prior = prior
        self.scale = scale
        self.sources = sources
        self.N = N
        self.M = M
        
        if likelihood == 'gauss':
            self.criterion = nn.MSELoss(reduction='sum')
        elif likelihood == 'laplace':
            self.criterion = LaplaceLoss()
        else:
            self.criterion = nn.BCELoss(reduction='sum')

        if alpha is None:
            self.alpha_pior = nn.Parameter(torch.ones(1,sources),requires_grad=False)
        else:
            self.alpha_prior = nn.Parameter(alpha,requires_grad=False)

    def forward(self, x, mu_theta_list, mu_phi_list, logvar_phi_list, s_list, pi_list, z_list, beta=1, scale=1, N=128, M=5):

        KLD_list = []
        for i in range(self.sources):
            if self.variational is True:
                if self.prior == 'laplace':
                    KLD_i = KLD_laplace(mu_phi_list[i],logvar_phi_list[i],scale=self.scale)
                    KLD_list.append(KLD_i)
                else:
                    KLD_i = KLD_gauss(mu_phi_list[i], logvar_phi_list[i]) 
                    KLD_list.append(KLD_i)
        
        ELL, pi_new_list, b_new, q_s_x = free_energy_first_term_and_pies_mnist_k_sources(x, scale, N, M, s_list, pi_list, mu_theta_list, z_list, self.sources)

        loss = -ELL + sum(beta * KLD for KLD in KLD_list)
        free_energy = -loss
        
        return loss, free_energy, ELL, KLD_list, pi_new_list, b_new, q_s_x
    
class Loss_MNIST_fixed_K(nn.Module):
    def __init__(self, sources=2, alpha=None, likelihood='bernoulli', variational=True, prior='gauss', scale=1.0, N=128, M=5):
        super(Loss_MNIST_fixed_K, self).__init__()
        self.variational = variational
        self.prior = prior
        self.scale = scale
        self.sources = sources
        self.N = N
        self.M = M
        
        if likelihood == 'gauss':
            self.criterion = nn.MSELoss(reduction='sum')
        elif likelihood == 'laplace':
            self.criterion = LaplaceLoss()
        else:
            self.criterion = nn.BCELoss(reduction='sum')

        if alpha is None:
            self.alpha_pior = nn.Parameter(torch.ones(1,sources),requires_grad=False)
        else:
            self.alpha_prior = nn.Parameter(alpha,requires_grad=False)

    def forward(self, x, mu_theta_list, mu_phi_list, logvar_phi_list, s_list, z_list, K, beta=1, scale=1, N=128, M=5):

        KLD_list = []
        for i in range(self.sources):
            if self.variational is True:
                if self.prior == 'laplace':
                    KLD_i = KLD_laplace(mu_phi_list[i],logvar_phi_list[i],scale=self.scale)
                    KLD_list.append(KLD_i)
                else:
                    KLD_i = KLD_gauss(mu_phi_list[i], logvar_phi_list[i])  # DKL [q(z_1|x) || p(z_1)]
                    KLD_list.append(KLD_i)
        
        ELL, b_new, q_s_x = free_energy_first_term_and_pies_mnist_fixed_K(x, 
                                                                        scale, 
                                                                        N, 
                                                                        M, 
                                                                        K,
                                                                        s_list, 
                                                                        mu_theta_list, 
                                                                        z_list, 
                                                                        self.sources)
        

        loss = -ELL + sum(beta * KLD for KLD in KLD_list)
        free_energy = -loss
        
        return loss, free_energy, ELL, KLD_list, b_new, q_s_x
    