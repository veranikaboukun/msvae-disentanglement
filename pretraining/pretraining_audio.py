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
# - Mildly modified original train.py from VAEM-BSS pretrain
#   individual VAE experts for audio.
# - Added saving of individual posterior statistics (mu_z and logvar_z).
# - Added custom plotting functions for individual analysis.
# - Removed the beta annealing procedure.
# - Updated script to support ECML PKDD 2026 paper contribution:
#   "Disentanglement in a Multi-Stream VAE"
# Original template repository: https://github.com
# ----------------------------------------------------------------------

import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from src.model import *
from src.model_streams import MultiStream_VAE_audio
from src.argparser_pretrain_audio import *
from src.dataload_audio import *
import glob
import librosa
from torch.utils.data import DataLoader


def train(epoch):

    model.train()
    train_losses = torch.zeros(3)

    for batch_idx, data in enumerate(train_loader):
        
        data = data.to(device).view(-1,dimx)

        optimizer.zero_grad()
        recon_y_list, mu_z_list, logvar_z_list, _ = model(data, M=args.m_samples) 
        recon_y = recon_y_list[0].squeeze(1)
        mu_z = mu_z_list[0]
        logvar_z = logvar_z_list[0]

        loss, ELL, KLD = loss_function(data, recon_y, mu_z, logvar_z, beta=beta) 
        loss.backward()
        optimizer.step()

        train_losses[0] += loss.item()/len(data)/args.batch_size
        train_losses[1] += ELL.item()/len(data)/args.batch_size
        train_losses[2] += KLD.item()/args.batch_size

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t -ELL: {:5.6f} \t KLD: {:5.6f} \t Loss: {:5.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                ELL.item()/len(data)/args.batch_size,KLD.item()/args.batch_size,loss.item()/len(data)/args.batch_size),flush=True)

    train_losses /= len(train_loader)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_losses[0]),flush=True)

    return train_losses

def test(epoch):

    model.eval()
    test_losses = torch.zeros(3)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            
            data = data.to(device).view(-1,dimx)

            recon_y_list, mu_z_list, logvar_z_list, _ = model(data, M=args.m_samples) 
            recon_y = recon_y_list[0].squeeze(1)
            mu_z = mu_z_list[0]
            logvar_z = logvar_z_list[0]
          
            loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, beta=beta)

            test_losses[0] += loss.item()/len(data)/args.batch_size
            test_losses[1] += ELL.item()/len(data)/args.batch_size
            test_losses[2] += KLD.item()/args.batch_size

        merge = ((epoch - 1) % args.save_reco_every) == 0
        if merge:
            n = min(data.size(0), 6)
            ncols = (1+args.sources)
            comparison = torch.zeros(n*ncols,1, f, t)
            comparison[::ncols] = data.view(data.size(0), 1, f, t)[:n]
            comparison[1::ncols] = recon_y.view(data.size(0), 1, f, t)[:n]
            
            _, axes = plt.subplots(n, ncols, figsize=(ncols * 2, n * 2))
            for k in range(n):
                n_label = 3
                # Plot original spectrogram
                ax1 = axes[k, 0]
                S_db = librosa.amplitude_to_db(np.abs(comparison[k * ncols].squeeze()), ref=np.max)
                librosa.display.specshow(S_db, sr=args.fs, hop_length=args.hopfrac, x_axis='time', y_axis='linear', ax=ax1)
                ax1.set_title("Original")
                ax1.set_ylim(0, 8000)
                ax1.label_outer()
                [l.set_visible(False) for (i,l) in enumerate(ax1.xaxis.get_ticklabels()) if i % n_label != 0]

                # Plot reconstructed mixture spectrogram
                ax2 = axes[k, 1]
                S_db = librosa.amplitude_to_db(np.abs(comparison[k * ncols + 1].squeeze()), ref=np.max)
                librosa.display.specshow(S_db, sr=args.fs, hop_length=args.hopfrac, x_axis='time', y_axis='linear', ax=ax2)
                ax2.set_title("Reconstructed")
                ax2.set_ylim(0, 8000)
                ax2.label_outer()
                [l.set_visible(False) for (i,l) in enumerate(ax2.xaxis.get_ticklabels()) if i % n_label != 0]
                
            plt.tight_layout()
            plt.savefig(f"{args.save_path}/reconstruction_{epoch}.png")
            plt.close()

        test_losses /= len(test_loader)
        print('====> Test set loss: {:.4f}'.format(test_losses[0]),flush=True)

    return test_losses

def plot_losses(losses):
	plt.figure()
	plt.plot(np.array(range(1,args.epochs+1)),losses["train"][:,0].view(-1),label="Train")
	plt.plot(np.array(range(1,args.epochs+1)),losses["test"][:,0].view(-1),label="Test")
	plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend(), plt.xlim(1,args.epochs)
	plt.savefig(args.save_path + '/losses.png')
	plt.close()


args = parser.parse_args()
torch.manual_seed(42)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

data_file, training_file = (
        args.save_path + "/images_labels.h5",
        args.save_path + "/training.h5",
    )

train_files_mix = sorted(glob.glob(args.train_dir + '*.wav', recursive=True))
n_total = len(train_files_mix)
n_subset = int(np.ceil(args.percent_labeled * n_total))

np.random.seed(args.seed)  
subset_indices = np.random.choice(n_total, n_subset, replace=False)

valid_files_mix = sorted(glob.glob(args.valid_dir + '*.wav', recursive=True))

spec_transform = Spectrogram(n_fft=args.nfft, 
                             win_length=args.winlen, 
                             hop_length=args.hopfrac, 
                             pad_mode='constant')

train_files_subset = [train_files_mix[i] for i in subset_indices]
train_dataset_mix = InstrumentSpectrogramDataset(train_files_subset, 
                                                 args.winlen, 
                                                 transform=spec_transform, 
                                                 max_timesteps=17,
                                                 transpose=False) 

test_dataset_mix = InstrumentSpectrogramDataset(valid_files_mix, 
                                                args.winlen, 
                                                transform=spec_transform, 
                                                max_timesteps=17, 
                                                transpose=False)


num_train = len(train_dataset_mix)
num_test = len(test_dataset_mix)

f, t = train_dataset_mix[0].shape
dimx = int(t*f)
D = dimx

train_loader = DataLoader(train_dataset_mix, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset_mix, batch_size=args.batch_size_test, shuffle=True)

model = MultiStream_VAE_audio(dimx=dimx, 
                                dimz=args.dimz, 
                                n_sources=args.sources, 
                                device=device, 
                                variational=args.variational).to(device)


loss_function = Loss(sources=args.sources,
                     likelihood='laplace',
                     variational=args.variational,
                     prior=args.prior,
                     scale=args.scale)

optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                      step_size=1, 
                                      gamma=args.decay, 
                                      last_epoch=-1)

losses = {"train": torch.zeros(args.epochs,3), "test": torch.zeros(args.epochs,3)}

for epoch in range(1, args.epochs+1):
    beta = min(1.0,(epoch)/min(args.epochs,args.warm_up)) * args.beta_max

    losses["train"][epoch-1] = train(epoch)
    losses["test"][epoch-1] = test(epoch)

    if optimizer.param_groups[0]['lr'] >= 1.0e-5:
        scheduler.step()

    with torch.no_grad():
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), args.save_path + 'model_'+('vae' if args.variational else 'ae')+'_K' + str(args.sources) +  '.pt')
            plot_losses(losses)


