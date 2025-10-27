import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from src.model import *
from src.model_streams import MultiStream_VAE
from src.argparser_pretrain import *
# from src.utils import *
from src.dataloader import *
from src.H5Logger import *

def train(epoch):

    model.train()
    train_losses = torch.zeros(3)

    for batch_idx, (data,_) in enumerate(train_loader):
        
        data = data.to(device).view(-1,dimx)

        optimizer.zero_grad()
        recon_y_list, mu_z_list, logvar_z_list, _ = model(data, M=args.m_samples) 
        recon_y = recon_y_list[0].squeeze(1)
        mu_z = mu_z_list[0]
        logvar_z = logvar_z_list[0]

        loss, ELL, KLD = loss_function(data, recon_y, mu_z, logvar_z, input_name="MNIST", beta=beta) 
        loss.backward()
        optimizer.step()

        train_losses[0] += loss.item()
        train_losses[1] += ELL.item()
        train_losses[2] += KLD.item()

        # if save 'labels': learned mu_z and logvar_z for every image
        if args.log_labels:
            if epoch == args.epochs:
                if data.shape == torch.Size([args.batch_size, dimx]):
                    to_log = {"data": data, "mu_z": mu_z, "logvar_z": logvar_z} 
                else:
                    to_log = {"data_last_batch": data, "mu_z_last_batch": mu_z, "logvar_z_last_batch": logvar_z} 

                if to_log is not None:
                    logger.append_and_write(**to_log)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t -ELL: {:5.6f} \t KLD: {:5.6f} \t Loss: {:5.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                ELL.item() / len(data),KLD.item() / len(data),loss.item() / len(data)))

    train_losses /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_losses[0]))

    return train_losses

def test(epoch):

    model.eval()
    test_losses = torch.zeros(3)

    with torch.no_grad():
        for i, (data,_) in enumerate(test_loader):
            
            data = data.to(device).view(-1,dimx)

            recon_y_list, mu_z_list, logvar_z_list, _ = model(data, M=args.m_samples) 
            recon_y = recon_y_list[0].squeeze(1)
            mu_z = mu_z_list[0]
            logvar_z = logvar_z_list[0]
          
            loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, input_name="MNIST", beta=beta)

            test_losses[0] += loss.item()
            test_losses[1] += ELL.item()
            test_losses[2] += KLD.item()

        merge = ((epoch - 1) % args.save_reco_every) == 0
        if merge:
            n = min(data.size(0), 6)
            ncols = (1+args.sources)
            comparison = torch.zeros(n*ncols,1,28,28)
            comparison[::ncols] = data.view(data.size(0), 1, 28, 28)[:n]
            comparison[1::ncols] = recon_y.view(data.size(0), 1, 28, 28)[:n]

            fig, axes = plt.subplots(n, ncols, figsize=(ncols * 2, n * 2))
            for k in range(n):
                # Plot original image
                ax = axes[k, 0]
                ax.imshow(comparison[k * ncols].squeeze(), cmap='gray')
                ax.set_title("Original")
                ax.axis('off')

                # Plot reconstructed mixture image
                ax = axes[k, 1]
                ax.imshow(comparison[k * ncols + 1].squeeze(), cmap='gray')
                ax.set_title("Reconstructed")
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(f"{args.save_path}/reconstruction_{epoch}.png")
            plt.close()

        test_losses /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_losses[0]))

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
print(torch.initial_seed())

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

data_file, training_file = (
        args.save_path + "/images_labels.h5",
        args.save_path + "/training.h5",
    )

logger = H5Logger(data_file)


subset = [args.subset]


train_loader, test_loader = get_subset_dataloaders_and_indices(args.data_directory, 
                                                               args.batch_size, 
                                                               subset, 
                                                               args.percent_labeled, # control here the % of supervision
                                                               args.save_path, kwargs)


dimx = int(28*28)
D = dimx

model = MultiStream_VAE(dimx=dimx, 
                        dimz=args.dimz, 
                        n_sources=args.sources, 
                        device=device, 
                        variational=args.variational, 
                        freeze_decoder=args.freeze_decoder).to(device)

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

losses = {"train": torch.zeros(args.epochs,3), 
          "test": torch.zeros(args.epochs,3)}

for epoch in range(1, args.epochs+1):
    beta = 1 # no annealing
    losses["train"][epoch-1] = train(epoch)
    losses["test"][epoch-1] = test(epoch)

    if optimizer.param_groups[0]['lr'] >= 1.0e-5:
        scheduler.step()

    with torch.no_grad():
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), args.save_path + 'model_'+('vae' if args.variational else 'ae')+'_K' + str(args.sources) +  '.pt')

plot_losses(losses)
