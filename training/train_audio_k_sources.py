# Copyright (C) 2021 Julian Neri, Roland Badeau, Philippe Depalle
# Copyright (C) 2026 Veranika Boukun <veranika.boukun@uni-oldenburg.de>
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
# Modifications by Veranika Boukun (2026):
# - Heavily modified original train.py from VAEM-BSS for audio.
# - Integrated all necessary pre-processing and EM steps.
# - Added entropy and accuracy calculations.
# - Added multiple custom plotting functions for individual analysis.
# - Removed the beta annealing procedure.
# - Updated script to support ECML PKDD 2026 paper contribution:
#   "Disentanglement in a Multi-Stream VAE"
# Original template repository: https://github.com
# ----------------------------------------------------------------------

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch import nn, optim
import matplotlib.pyplot as plt
from src.model_streams import MultiStream_VAE_audio, Loss_k_sources
from src.argparser_audio import parser
from src.utils import compute_pi_from_labels, accuracy_score, get_ELBO_entropies_k_sources
from src.dataloader import *
import itertools
from src.dataload_audio import *
import glob
import pandas as pd
from torch.distributions import Laplace
import librosa
from torch.utils.data import DataLoader


def train(epoch, b):

    model.train()
    train_losses = torch.zeros(3+args.sources)
    batch_parameters = torch.zeros(1+args.sources)
    entropies = torch.zeros(5+args.sources)
    
    for batch_idx, (data, _) in enumerate(spectrogram_dataloader_train):  
        
        data = data.to(device).view(-1, dimx)
        data = data.float()
        optimizer_all.zero_grad()

        mu_theta_list, mu_phi_list, logvar_phi_list, z_list = model(data, 
                                                                    M=args.m_samples) 
        
        torch.autograd.set_detect_anomaly(True)
    
        loss, free_energy, ELL, KLD_list, pi_new_list, b_new, q_s_x = loss_overall(data, 
                                                                                    mu_theta_list, 
                                                                                    mu_phi_list, 
                                                                                    logvar_phi_list, 
                                                                                    s_list, 
                                                                                    pi_init_values, 
                                                                                    z_list, 
                                                                                    beta=beta, 
                                                                                    scale=b, 
                                                                                    N=args.batch_size, 
                                                                                    M=args.m_samples) 
        
        H_enc_list, H_dec, H_s_posterior, H_s_prior, H_z_prior, H_sum = get_ELBO_entropies_k_sources(data, 
                                                                                                     mu_phi_list, 
                                                                                                     logvar_phi_list, 
                                                                                                     b, 
                                                                                                     s_list, 
                                                                                                     pi_init_values, 
                                                                                                     args.dimz, 
                                                                                                     q_s_x, 
                                                                                                     device, 
                                                                                                     args.sources)
        
        loss.backward()
        optimizer_all.step()
        
        train_losses[0] += loss.item()
        train_losses[1] += ELL.item()
        train_losses[2] += free_energy.item() 
    
        for i, KLD in enumerate(KLD_list):
            train_losses[3 + i] += KLD.item()
        
        batch_parameters[0] += b_new.item() 
        for i, pi_new in enumerate(pi_new_list):
            batch_parameters[1 + i] += pi_new.item()
            
        entropies[0] += H_dec.item()
        entropies[1] += H_s_posterior.item()
        entropies[2] += H_s_prior.item()
        entropies[3] += H_z_prior.item()
        entropies[4] += H_sum.item()
        for i, H_enc in enumerate(H_enc_list):
            entropies[5 + i] += H_enc.item()

        args.log_interval = 1
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t -ELL: {:5.6f} \t Loss: {:5.6f}'.format(
                epoch, batch_idx * args.batch_size, len(spectrogram_dataloader_train.dataset),
                100. * batch_idx / len(spectrogram_dataloader_train),
                ELL.item(), loss.item()), flush=True)
                
    train_losses /= len(spectrogram_dataloader_train)
    batch_parameters /= len(spectrogram_dataloader_train)
    entropies /= len(spectrogram_dataloader_train)
    
    if pi_update_fixed:
        for i in range(args.sources):
            pi_list[i] = pi_init_values[i]

    else:
        for i in range(args.sources):
            pi_list[i] = batch_parameters[1+i]
        
    b = batch_parameters[0]  
    
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_losses[0]), flush=True)
    
    to_log = {"train_loss": train_losses[0], 
              "first_term_ELBO": train_losses[1], 
              "free_energy": train_losses[2], 
              "DKL_terms": train_losses[3:len(train_losses)],
              "b": b, 
              "pi_vector": batch_parameters[1:len(batch_parameters)],
              "H_dec": entropies[0], 
              "H_s_posterior": entropies[1], 
              "H_s_prior": entropies[2], 
              "H_z_prior": entropies[3],
              "H_sum": entropies[4],
              "H_enc_terms": entropies[5:len(entropies)]} # log the data by your preferred method

    return train_losses, pi_list, b, entropies 
 
def test(epoch, pi_list, b): 

    model.eval()
    test_losses = torch.zeros(2+args.sources)
    accuracies = torch.zeros(1)

    with torch.no_grad():
        for i, (data, labels) in enumerate(spectrogram_dataloader_test):
           
            data = data.to(device).view(-1, dimx)
            data = data.float()

            mu_theta_list, mu_phi_list, logvar_phi_list, z_list = model(data, M=args.m_samples)
            if epoch==1:
                loss, _, ELL, KLD_list, _, _, q_s_x = loss_overall(data, 
                                                                    mu_theta_list, 
                                                                    mu_phi_list, 
                                                                    logvar_phi_list, 
                                                                    s_list, 
                                                                    pi_init_values, 
                                                                    z_list, 
                                                                    beta=beta, 
                                                                    N=args.batch_size, 
                                                                    M=args.m_samples, 
                                                                    scale=b_init)

            else:
                loss, _, ELL, KLD_list, _, _, q_s_x = loss_overall(data, 
                                                                    mu_theta_list, 
                                                                    mu_phi_list, 
                                                                    logvar_phi_list, 
                                                                    s_list, 
                                                                    pi_list, 
                                                                    z_list, 
                                                                    beta=beta, 
                                                                    N=args.batch_size, 
                                                                    M=args.m_samples, 
                                                                    scale=b)

            test_losses[0] += loss.item()
            test_losses[1] += ELL.item()
            for i, KLD in enumerate(KLD_list):
                test_losses[2 + i] += KLD.item()

            labels_instruments = labels[:, 1:]
            _, target = torch.max(s_list.detach().cpu() @ labels_instruments.T.long(), dim=0) 
            _, prediction = torch.max(q_s_x.detach().cpu(), dim=1)

            accuracy = accuracy_score(target.squeeze(0), prediction)

            accuracies[0] += round(accuracy, 2)

        case_numbers = target 
        active_signals = []
        for i in range(labels_instruments.shape[0]):
            signal = torch.nonzero(labels_instruments[i, :]).squeeze(1)
            if signal.numel() == 0:
                active_signals.append(None)
            else:
                active_signals.append([x.item() for x in signal])

        max_values, max_indices = torch.max(q_s_x.detach().cpu(), dim=1)

        mu_theta_i_list = []
        for i in range(args.sources):
            mu_theta_i = mu_theta_list[i].sum(1).detach().cpu() 
            mu_theta_i_list.append(mu_theta_i)

        mu_theta_i_new_list = []
        for i, mu_theta_i in enumerate(mu_theta_i_list):
            mu_theta_i_new = torch.zeros_like(mu_theta_i).to(device)
            for n, index in enumerate(max_indices):
                s_vector = s_list[index].detach().cpu().float() 
                s_i = s_vector[i]
                mu_theta_i_new[n, ...] = mu_theta_i[n, ...] * s_i  

            mu_theta_i_new_list.append(mu_theta_i_new)

        recon_y = torch.stack(mu_theta_i_new_list, dim=2)
        recons = recon_y.sum(2)

        n = min(data.size(0), 8)
        ncols = (2+args.sources)

        comparison = torch.zeros(n*ncols,1, f, t)
        comparison[::ncols] = data.view(data.size(0), 1, f, t)[:n]
        comparison[1::ncols] = recons.view(data.size(0), 1, f, t)[:n]

        for i in range(args.sources):
            comparison[(i+2)::ncols] = recon_y[...,i].view(data.size(0), 1, f, t)[:n]

        merge = ((epoch - 1) % args.save_reco_every) == 0
        if merge:
            _, axes = plt.subplots(n, ncols, figsize=(ncols * 2, n * 2))
            for k in range(n):
                case_number=case_numbers[k].squeeze()
                n_label = 3
                ax1 = axes[k, 0]
                S_db = librosa.amplitude_to_db(np.abs(comparison[k * ncols].squeeze()), ref=np.max)
                librosa.display.specshow(S_db, 
                                         sr=args.fs, 
                                         hop_length=args.hopfrac, 
                                         x_axis='time', 
                                         y_axis='linear', 
                                         ax=ax1)
                ax1.set_title(f"Original, \n Ground Truth Case {case_number} \n Speakers: {active_signals[k]}", fontsize=6)
                ax1.set_ylim(0, 8000)
                ax1.label_outer()
                [l.set_visible(False) for (i,l) in enumerate(ax1.xaxis.get_ticklabels()) if i % n_label != 0]

                ax2 = axes[k, 1]
                S_db = librosa.amplitude_to_db(np.abs(comparison[k * ncols + 1].squeeze()), ref=np.max)
                librosa.display.specshow(S_db, 
                                         sr=args.fs, 
                                         hop_length=args.hopfrac, 
                                         x_axis='time', 
                                         y_axis='linear', 
                                         ax=ax2)
                ax2.set_title("Reconstructed", fontsize=6)
                ax2.set_ylim(0, 8000)
                ax2.label_outer()
                [l.set_visible(False) for (i,l) in enumerate(ax2.xaxis.get_ticklabels()) if i % n_label != 0]

                for j in range(args.sources):
                    ax = axes[k, j + 2]
                    S_db = librosa.amplitude_to_db(np.abs(comparison[k * ncols + j + 2].squeeze()), ref=np.max)
                    librosa.display.specshow(S_db, 
                                             sr=args.fs, 
                                             hop_length=args.hopfrac, 
                                             x_axis='time', 
                                             y_axis='linear', 
                                             ax=ax)
                    ax.set_title(f"Source {j+1}, \n Case {max_indices[k]}, Sure \n{format(max_values[k]*100, '.1f')}%", fontsize=6)
                    ax.set_ylim(0, 8000)
                    ax.label_outer()
                    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n_label != 0]

            plt.tight_layout()
            plt.savefig(f"{args.save_path}/reconstruction_{epoch}.pdf")
            plt.close()

        test_losses /= len(spectrogram_dataloader_test)
        print('====> Test set loss: {:.4f}'.format(test_losses[0]), flush=True)

        accuracies /= len(spectrogram_dataloader_test)
        print('====> Test set accuracy: {:.4f}'.format(accuracies[0]), flush=True)

        to_log = {"test_loss": test_losses[0], "test_accuracy": accuracies[0]} # log with your preferred method

    return test_losses, accuracies

def plot_losses(losses):
	plt.figure()
	plt.plot(np.array(range(1,args.epochs+1)),losses["train"][:,0].view(-1),label="Train")
	plt.plot(np.array(range(1,args.epochs+1)),losses["test"][:,0].view(-1),label="Test")
	plt.xlabel('Epoch'), plt.ylabel('-ELBO'), plt.legend(), plt.xlim(1,args.epochs)
	plt.savefig(args.save_path + '/losses.png')
	plt.close()

def plot_entropy_decoder(entropies):
    plt.figure()
    plt.plot(np.array(range(1,args.epochs+1)),entropies["train"][:,0].view(-1),label="H_dec")
    plt.xlabel('Epoch'), plt.ylabel('Average Entropy'), plt.legend(), plt.xlim(1,args.epochs)
    plt.savefig(args.save_path + '/entropy_decoder.png')
    plt.close()

def plot_entropy_encoders(entropies):
    plt.figure()
    for i in range(args.sources):
        plt.plot(np.array(range(1,args.epochs+1)),entropies["train"][:,5+i].view(-1),label="H_enc_"+str(i))
    plt.xlabel('Epoch'), plt.ylabel('Average Entropy'), plt.legend(), plt.xlim(1,args.epochs)
    plt.savefig(args.save_path + '/entropy_encoders.png')
    plt.close()

def plot_entropy_priors(entropies):
    plt.figure()
    plt.plot(np.array(range(1,args.epochs+1)),entropies["train"][:,2].view(-1),label="H_s_prior")
    plt.plot(np.array(range(1,args.epochs+1)),entropies["train"][:,3].view(-1),label="H_z_prior")
    plt.xlabel('Epoch'), plt.ylabel('Average Entropy'), plt.legend(), plt.xlim(1,args.epochs)
    plt.savefig(args.save_path + '/entropy_priors.png')
    plt.close()

def plot_entropy_s_posterior(entropies):
    plt.figure()
    plt.plot(np.array(range(1,args.epochs+1)),entropies["train"][:,1].view(-1),label="H_s_posterior")
    plt.xlabel('Epoch'), plt.ylabel('Average Entropy'), plt.legend(), plt.xlim(1,args.epochs)
    plt.savefig(args.save_path + '/entropy_s_posterior.png')
    plt.close()

def plot_entropy_sum_with_ELBO(losses, entropies):
    plt.figure()
    plt.plot(np.array(range(1,args.epochs+1)),losses["train"][:,2].view(-1),label="ELBO")
    plt.plot(np.array(range(1,args.epochs+1)),entropies["train"][:,4].view(-1),label="H_sum")
    plt.xlabel('Epoch'), plt.ylabel('Average Entropy'), plt.legend(), plt.xlim(1,args.epochs)
    plt.savefig(args.save_path + '/entropy_sum_ELBO.png')
    plt.close()

def plot_accuracy(accuracies):
    plt.figure()
    plt.plot(np.array(range(1,args.epochs+1)),accuracies["test"][:,0].view(-1),label="Test")
    plt.xlabel('Epoch'), plt.ylabel('Average Prediction Accuracy, %'), plt.legend(), plt.xlim(1,args.epochs)
    plt.savefig(args.save_path + '/test_accuracy.png')
    plt.close()

args = parser.parse_args()
torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

data_file, training_file = (
        args.save_path + "/images_labels.h5",
        args.save_path + "/training.h5",
    )

valid_files_mix = sorted(glob.glob(args.test_dir_mixture + '*.wav', recursive=True))
labels_test = pd.read_csv(args.label_file_test).values

spec_transform = Spectrogram(n_fft=args.nfft, 
                             win_length=args.winlen, 
                             hop_length=args.hopfrac, 
                             pad_mode='constant')

 
test_dataset_mix = SpeechSpectrogramDataset(valid_files_mix, 
                                            args.winlen, 
                                            transform=spec_transform, 
                                            max_timesteps=17, 
                                            transpose=False)


# Replace with zeros those frames which are labeled with (0, 0)
test_data = []
for idx in range(len(test_dataset_mix)):
    spec = test_dataset_mix[idx]
    label = labels_test[idx]
    if args.sources == 2:
        if label[1] == 0 and label[2] == 0:
            mean = torch.zeros_like(spec)
            spec = Laplace(mean, args.scale).sample()

    elif args.sources == 3: 
        if label[1] == 0 and label[2] == 0 and label[3] == 0:
            mean = torch.zeros_like(spec)
            spec = Laplace(mean, args.scale).sample()
            
    test_data.append(spec)

test_data_tensor = torch.stack(test_data)

labels_test = torch.Tensor(labels_test)

test_set = TensorDataset(test_data_tensor, labels_test)
num_test = len(test_set)

labels_train = pd.read_csv(args.label_file_train).values
train_files_mix = sorted(glob.glob(args.train_dir_mixture + '*.wav', recursive=True))
train_dataset_mix = SpeechSpectrogramDataset(train_files_mix, 
                                            args.winlen, 
                                            transform=spec_transform, 
                                            max_timesteps=17,
                                            transpose=False) 

train_data = []
for idx in range(len(train_dataset_mix)):
    spec = train_dataset_mix[idx]
    label = labels_train[idx]
    if args.sources == 2:
        if label[1] == 0 and label[2] == 0:
            mean = torch.zeros_like(spec)
            spec = Laplace(mean, args.scale).sample()
    elif args.sources == 3: 
        if label[1] == 0 and label[2] == 0 and label[3] == 0:
            mean = torch.zeros_like(spec)
            spec = Laplace(mean, args.scale).sample()

    train_data.append(spec)

train_data_tensor = torch.stack(train_data)

labels_train = torch.Tensor(labels_train)
train_set = TensorDataset(train_data_tensor, labels_train)
num_train = len(train_set)

resulting_activations_test = compute_pi_from_labels(args.label_file_test)
pi_test_values = [round(resulting_activations_test[i],2) for i in range(len(resulting_activations_test)-1)]
pi_test_str = ", ".join(str(pi) for pi in pi_test_values)
print("Approximate Bernoulli (pi) test parameters: " + pi_test_str)

num_train = len(train_set)

print('Number of training datapoints: ' + str(num_train))
print('Number of testing datapoints: ' + str(num_test))

f, t = train_set[0][0].shape
dimx = int(t*f)
D = dimx

# dataloader
spectrogram_dataloader_train = DataLoader(train_set, 
                                          batch_size=args.batch_size, 
                                          shuffle=True)
spectrogram_dataloader_test = DataLoader(test_set, 
                                         batch_size=args.batch_size, 
                                         shuffle=True)

H = args.sources
s_list = torch.asarray(list(itertools.product(range(2), repeat=H))).to(device)

# get generative parameters: pi and scale b
resulting_activations_train = compute_pi_from_labels(args.label_file_train)
pi_train_values = [round(resulting_activations_train[i],2) for i in range(len(resulting_activations_train)-1)]
pi_train_str = ", ".join(str(pi) for pi in pi_train_values)
print("Approximate Bernoulli (pi) train parameters: " + pi_train_str)

print("Generative Bernoulli (pi) train parameters: " + str(pi_train_values))

pi_update_fixed = args.pi_update_fixed

print("Generative noise scale (b): " + str(args.scale))

pi_init_values = []

for i in range(args.sources):
    pi_init_key = f'pi_{i}_init'
    pi_init_value = getattr(args, pi_init_key)
    pi_init_values.append(pi_init_value)

pi_init_str = ", ".join(str(pi) for pi in pi_init_values)
pi_list = pi_init_values
print("Initial Bernoulli (pi) parameters: " + pi_init_str) 

b = args.scale_init
b_init = args.scale_init
print("Initial noise scale (b): " + str(b))

# == Initialize desired MS-VAE model ====
model = MultiStream_VAE_audio(dimx=dimx, 
                              dimz=args.dimz, 
                              n_sources=args.sources, 
                              device=device, 
                              variational=args.variational, 
                              freeze_decoder=args.freeze_decoder)

print("Freeze decoder: " + str(args.freeze_decoder))
print('Decoder unfreeze after N epochs: ' + str(args.unfreeze_decoders_after_n))

# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    
model.to(device)

# Log initial and generative model parameters, dimension of s
to_log = {"pi_init_list": torch.tensor(pi_init_values), 
          "pi_train_list": torch.tensor(pi_train_values),
          "pi_test_list": torch.tensor(pi_test_values),
          "b_init": torch.tensor(b), 
          "s_dim": torch.tensor(H)} # log data with your preferred method

vae_list = []

pretrained_models = [
    getattr(args, f'pretrained_speaker_{i}_model')
    for i in range(args.sources)
]

for i in range(args.sources):
    vae = MultiStream_VAE_audio(dimx=dimx, 
                                dimz=args.dimz, 
                                n_sources=1, 
                                device='cpu', 
                                variational=True)
    
    vae.load_state_dict(torch.load(pretrained_models[i]))  # Load pretrained models (only decoders will be used)
    vae_list.append(vae)

if args.freeze_decoder:
    for i in range(args.sources):
        model.module.Decoders[i].load_state_dict(vae_list[i].Decoders[0].state_dict()) # sometimes "module" is required sometimes not - adjust as necessary!

loss_overall = Loss_k_sources(sources=args.sources, 
                               likelihood='laplace', 
                               variational=args.variational, 
                               prior=args.prior) 

optimizer_all = optim.Adam(model.parameters(), lr = args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer_all, step_size=1, gamma=args.decay, last_epoch=-1)

# Losses and entropies are logged to dictionaries while training
losses = {"train": torch.zeros(args.epochs,3+args.sources), "test": torch.zeros(args.epochs,2+args.sources)}
entropies = {"train": torch.zeros(args.epochs,5+args.sources)}
accuracies = {"test": torch.zeros(args.epochs,1)}

for epoch in range(1, args.epochs+1):
    beta = 1.0 
    if args.unfreeze_decoders_after_n:
        if epoch == args.unfreeze_n:
            print("Unfreeze at epoch: " + str(args.unfreeze_n))
            print("Before unfreeze: ", optimizer_all.param_groups[0]['params'][0].requires_grad)
            learning_rate_e_n = optimizer_all.param_groups[0]['lr']
            model.unfreeze_decoder()
            optimizer_all = torch.optim.Adam(model.parameters(), lr = learning_rate_e_n)
            print('Decoder weights will be now updated', flush=True)
            print("After unfreeze: ", optimizer_all.param_groups[0]['params'][0].requires_grad)

    losses["train"][epoch-1], pi_list, b, entropies["train"][epoch-1] = train(epoch, b)
    losses["test"][epoch-1], accuracies["test"][epoch-1] = test(epoch, pi_list, b)

    if optimizer_all.param_groups[0]['lr'] >= 1.0e-5:
        scheduler.step()

    with torch.no_grad():
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), args.save_path + '/model_'+('vae' if args.variational else 'ae')+'_K' + str(args.sources) +  '.pt')

plot_losses(losses)
plot_entropy_decoder(entropies)
plot_entropy_encoders(entropies)
plot_entropy_priors(entropies)
plot_entropy_s_posterior(entropies)
plot_entropy_sum_with_ELBO(losses, entropies)
plot_accuracy(accuracies)
