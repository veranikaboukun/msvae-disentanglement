'''
Neri, R. Badeau, P. Depalle, “Unsupervised Blind Source Separation with Variational Auto-Encoders”,
29th European Signal Processing Conference (EUSIPCO), Dublin, Ireland, August 2021.
Code available at: https://github.com/jundsp/VAE-BSS"
'''
import torch
import numpy as np

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

