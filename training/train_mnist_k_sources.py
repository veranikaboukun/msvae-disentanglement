import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
from src.model_streams import *
from src.argparser_mnist import *
from src.utils import *
from src.dataloader import *
import itertools
from src.H5Logger import *
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

def train(epoch, b):

    model.train()
    train_losses = torch.zeros(3+args.sources)
    batch_parameters = torch.zeros(1+args.sources)
    entropies = torch.zeros(5+args.sources)

    for batch_idx, (data, _) in enumerate(mnist_dataloader_train):
        
        data = data.to(device).view(-1, args.dimx)
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
                                                                                                    q_s_x, device, 
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
                epoch, batch_idx * args.batch_size, len(mnist_dataloader_train.dataset),
                100. * batch_idx / len(mnist_dataloader_train),
                ELL.item(), loss.item()), flush=True)
                
    
    train_losses /= len(mnist_dataloader_train)
    batch_parameters /= len(mnist_dataloader_train)
    entropies /= len(mnist_dataloader_train)
    
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
              "H_enc_terms": entropies[5:len(entropies)]}
    if to_log is not None:
        logger.append_and_write(**to_log)

    return train_losses, pi_list, b, entropies 

def test(epoch, pi_list, b): 

    model.eval()
    test_losses = torch.zeros(2+args.sources)
    metrics = torch.zeros(2)

    with torch.no_grad():
        for i, (data, labels) in enumerate(mnist_dataloader_test):
           
            data = data.to(device).view(-1, args.dimx)
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

            labels_s_list = s_list.detach().cpu() @ labels.T.long()
            _, target = torch.max(labels_s_list, dim=1) 
            _, prediction = torch.max(q_s_x.detach().cpu(), dim=1)

            accuracy = accuracy_score(target.squeeze(0), prediction)

            metrics[0] += round(accuracy, 2)
        
        case_numbers = target 
        active_signals = []
        for i in range(labels.shape[0]):
            signal = torch.nonzero(labels[i, :, 0]).squeeze(1)
            if signal.numel() == 0:
                active_signals.append(None)
            else:
                active_signals.append([x.item() for x in signal])

        # Given highest q_s_x, multiply means by the corresponding s's
        max_values, max_indices = torch.max(q_s_x.detach().cpu(), dim=1)

        # Find maximum number of activations in the mini-batch: matches with visual observations
        activations_per_vector = []
        # Determine the number of 1s in the selected combinations
        for index in max_indices:
            # Retrieve the binary vector for this index
            selected_combination = s_list[index]
            num_activations = selected_combination.sum().item()
            activations_per_vector.append(num_activations)

        mu_theta_i_list = []
        for i in range(args.sources):
            mu_theta_i = mu_theta_list[i].sum(1).detach().cpu()
            mu_theta_i_list.append(mu_theta_i)
        
        mu_theta_i_new_list = []
        for i, mu_theta_i in enumerate(mu_theta_i_list):
            # Result tensor for activated mean images
            mu_theta_i_new = torch.zeros_like(mu_theta_i).to(device)
            for n, index in enumerate(max_indices):
                # Predict s_vector to apply based on model
                s_vector = s_list[index].detach().cpu().float() 
                s_i = s_vector[i]
                # Use s_vector to scale (multiply) each image
                mu_theta_i_new[n, ...] = mu_theta_i[n, ...] * s_i  

            mu_theta_i_new_list.append(mu_theta_i_new)
            # to check look at s_vector, then let say check digit 10, 
            # i.e., mu_theta_i_new_list[9]
            # and let's say it appears in data points 0, 4, 
            # then mu_theta_i_new_list[9][0] and mu_theta_i_new_list[9][4] should be non-zero!

        recon_y = torch.stack(mu_theta_i_new_list, dim=2) 
        recons = recon_y.sum(2) 

        n = min(data.size(0), 8)
        ncols = (2+args.sources) 
        comparison = torch.zeros(n*ncols,1,28,28)
        comparison[::ncols] = data.view(data.size(0), 1, 28, 28)[:n]
        comparison[1::ncols] = recons.view(data.size(0), 1, 28, 28)[:n]

        for i in range(args.sources):
            comparison[(i+2)::ncols] = recon_y[...,i].view(data.size(0), 1, 28, 28)[:n]

        merge = ((epoch - 1) % args.save_reco_every) == 0
        if merge:

            _, axes = plt.subplots(n, ncols, figsize=(ncols * 2, n * 2))
            for k in range(n):
                case_number=case_numbers[:, k].squeeze()
                # Plot original image
                ax = axes[k, 0]
                ax.imshow(comparison[k * ncols].squeeze(), cmap='gray')
                ax.set_title(f"Original, \n Ground Truth Case {case_number} \n Digits: {active_signals[k]}", fontsize=6) #TODO: error here with k 
                ax.axis('off')

                # Plot reconstructed mixture image
                ax = axes[k, 1]
                ax.imshow(comparison[k * ncols + 1].squeeze(), cmap='gray')
                ax.set_title("Reconstructed",fontsize=6)
                ax.axis('off')

                # Plot reconstructed images for each source with case and q_s_x% annotations
                for j in range(args.sources):
                    ax = axes[k, j + 2]
                    ax.imshow(comparison[k * ncols + j + 2].squeeze(), cmap='gray')
                    ax.set_title(f"Source {j+1}, \n Case {max_indices[k]}, Sure \n{format(max_values[k]*100, '.1f')}%", fontsize=6)
                    ax.axis('off')

            plt.tight_layout()
            plt.savefig(f"{args.save_path}/reconstruction_{epoch}.pdf")
            plt.close()

    
        test_losses /= len(mnist_dataloader_test)
        print('====> Test set loss: {:.4f}'.format(test_losses[0]), flush=True)

        metrics /= len(mnist_dataloader_test)
        print('====> Test accuracy: {:.4f}'.format(metrics[0]), flush=True)
        print('====> Test MIG score: {:.4f}'.format(metrics[1]), flush=True)

        to_log = {"test_loss": test_losses[0], "accuracy": metrics[0]}
        if to_log is not None:
            logger.append_and_write(**to_log)

    return test_losses, metrics

def plot_losses(losses):
	plt.figure()
	plt.plot(np.array(range(1,args.epochs+1)),losses["train"][:,0].view(-1),label="Train")
	plt.plot(np.array(range(1,args.epochs+1)),losses["test"][:,0].view(-1),label="Test")
	plt.xlabel('Epoch'), plt.ylabel('-ELBO'), plt.legend(), plt.xlim(1,args.epochs)
	plt.savefig(args.save_path + 'losses.png')
	plt.close()

def plot_entropy_decoder(entropies):
    plt.figure()
    plt.plot(np.array(range(1,args.epochs+1)),entropies["train"][:,0].view(-1),label="H_dec")
    plt.xlabel('Epoch'), plt.ylabel('Average Entropy'), plt.legend(), plt.xlim(1,args.epochs)
    plt.savefig(args.save_path + 'entropy_decoder.png')
    plt.close()

def plot_entropy_encoders(entropies):
    plt.figure()
    for i in range(args.sources):
        plt.plot(np.array(range(1,args.epochs+1)),entropies["train"][:,5+i].view(-1),label="H_enc_"+str(i))
    plt.xlabel('Epoch'), plt.ylabel('Average Entropy'), plt.legend(), plt.xlim(1,args.epochs)
    plt.savefig(args.save_path + 'entropy_encoders.png')
    plt.close()

def plot_entropy_priors(entropies):
    plt.figure()
    plt.plot(np.array(range(1,args.epochs+1)),entropies["train"][:,2].view(-1),label="H_s_prior")
    plt.plot(np.array(range(1,args.epochs+1)),entropies["train"][:,3].view(-1),label="H_z_prior")
    plt.xlabel('Epoch'), plt.ylabel('Average Entropy'), plt.legend(), plt.xlim(1,args.epochs)
    plt.savefig(args.save_path + 'entropy_priors.png')
    plt.close()

def plot_entropy_s_posterior(entropies):
    plt.figure()
    plt.plot(np.array(range(1,args.epochs+1)),entropies["train"][:,1].view(-1),label="H_s_posterior")
    plt.xlabel('Epoch'), plt.ylabel('Average Entropy'), plt.legend(), plt.xlim(1,args.epochs)
    plt.savefig(args.save_path + 'entropy_s_posterior.png')
    plt.close()

def plot_entropy_sum_with_ELBO(losses, entropies):
    plt.figure()
    plt.plot(np.array(range(1,args.epochs+1)),losses["train"][:,2].view(-1),label="ELBO")
    plt.plot(np.array(range(1,args.epochs+1)),entropies["train"][:,4].view(-1),label="H_sum")
    plt.xlabel('Epoch'), plt.ylabel('Average Entropy'), plt.legend(), plt.xlim(1,args.epochs)
    plt.savefig(args.save_path + 'entropy_sum_ELBO.png')
    plt.close()

def plot_accuracy(metrics):
    plt.figure()
    plt.plot(np.array(range(1,args.epochs+1)), metrics["test"][:,0].view(-1),label="Test")
    plt.xlabel('Epoch'), plt.ylabel('Average Prediction Accuracy, %'), plt.legend(), plt.xlim(1,args.epochs)
    plt.savefig(args.save_path + 'test_accuracy.png')
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

logger = H5Logger(training_file)

train_set = datasets.MNIST(args.data_directory, 
                           train=True, 
                           download=True, 
                           transform=transforms.ToTensor())

test_set = datasets.MNIST(args.data_directory, 
                          train=False, 
                          download=True, 
                          transform=transforms.ToTensor())

combined_data_list = []
valid_subset_list = []
pi_gen_values = []
pi_init_values = []

# Initialize dictionaries to store the training and testing indices
training_indices = {}
testing_indices = {}

for i in range(args.sources):

    training_key = f'training_indices_{i}'
    testing_key = f'testing_indices_{i}'
    
    # Load the npy files using dynamic names
    training_indices[i] = np.load(getattr(args, training_key))
    testing_indices[i] = np.load(getattr(args, testing_key))
    # Access training and testing indices from the dictionaries
    npy_training_indices = training_indices[i]
    npy_testing_indices = testing_indices[i]
    
    # Process the training data
    data_with_label = Subset(train_set, npy_training_indices)
    combined_data = ImageOnlyDataset(data_with_label)
    combined_data_list.append(combined_data)  
    
    # Process the testing/validation data
    valid_with_label = Subset(test_set, npy_testing_indices)
    valid_subset = ImageOnlyDataset(valid_with_label)
    valid_subset_list.append(valid_subset) 

    pi_gen_key = f'pi_{i}_gen'
    pi_gen_value = getattr(args, pi_gen_key)
    pi_gen_values.append(pi_gen_value)

    pi_init_key = f'pi_{i}_init'
    pi_init_value = getattr(args, pi_init_key)
    pi_init_values.append(pi_init_value)

pi_gen_str = ", ".join(str(pi) for pi in pi_gen_values)
pi_init_str = ", ".join(str(pi) for pi in pi_init_values)
print("Generative Bernoulli (pi) parameters: " + pi_gen_str)
print("Initial Bernoulli (pi) parameters: " + pi_init_str) 

num_samples_train = args.n_train 
num_samples_test = args.n_test 
print('Number of generated training datapoints: ' + str(num_samples_train))
print('Number of generated testing datapoints: ' + str(num_samples_test))

b_gen=args.scale
print("Generative noise scale (b): " + str(b_gen))

mnist_train_set = mix_mnist_data_bernoulli_arbitrary(combined_data_list, 
                                                     num_samples_train, 
                                                     pi_gen_values, b_gen)

mnist_test_set = mix_mnist_data_bernoulli_arbitrary(valid_subset_list, 
                                                     num_samples_test, 
                                                     pi_gen_values, b_gen)

mnist_dataloader_train = DataLoader(mnist_train_set, 
                                    batch_size=args.batch_size, 
                                    shuffle=True)

mnist_dataloader_test = DataLoader(mnist_test_set, 
                                   batch_size=args.batch_size, 
                                   shuffle=True)

H = args.H_dim 

s_list = torch.asarray(list(itertools.product(range(2), repeat=H))).to(device)

pi_update_fixed = args.pi_update_fixed

b = args.scale_init
b_init = args.scale_init
print("Initial noise scale (b): " + str(b))

pi_list = pi_init_values

model = MultiStream_VAE(dimx=args.dimx, 
                        dimz=args.dimz, 
                        n_sources=args.sources, 
                        device=device, 
                        variational=args.variational, 
                        freeze_decoder=args.freeze_decoder, 
                        freeze_encoder=args.freeze_encoder).to(device)

print("Freeze decoder: " + str(args.freeze_decoder))
print("Freeze encoder: " + str(args.freeze_encoder))

to_log = {"pi_init_list": torch.tensor(pi_init_values), 
          "pi_gen_list": torch.tensor(pi_gen_values),
          "b_init": torch.tensor(b), 
          "b_gen": torch.tensor(b_gen), 
          "s_dim": torch.tensor(H)} 

if to_log is not None:
    logger.append_and_write(**to_log)

vae_list = []
pretrained_models = [args.pretrained_0_model, 
                     args.pretrained_1_model, 
                     args.pretrained_2_model,
                     args.pretrained_3_model, 
                     args.pretrained_4_model, 
                     args.pretrained_5_model,
                     args.pretrained_6_model, 
                     args.pretrained_7_model, 
                     args.pretrained_8_model, 
                     args.pretrained_9_model]

print(pretrained_models)

for i in range(args.sources):
    vae = MultiStream_VAE(dimx=args.dimx, dimz=args.dimz, n_sources=1, device='cpu', variational=True)
    vae.load_state_dict(torch.load(pretrained_models[i]))  
    vae_list.append(vae)

if args.freeze_encoder:
    for i in range(args.sources):
        model.Encoders[i].load_state_dict(vae_list[i].Encoders[0].state_dict())

if args.freeze_decoder:
    for i in range(args.sources):
        model.Decoders[i].load_state_dict(vae_list[i].Decoders[0].state_dict())

loss_overall = Loss_k_sources(sources=args.sources, 
                              likelihood='laplace', 
                              variational=args.variational, 
                              prior=args.prior)  

optimizer_all = optim.Adam(model.parameters(), 
                           lr = args.learning_rate)

losses = {"train": torch.zeros(args.epochs,3+args.sources), "test": torch.zeros(args.epochs,2+args.sources)}
entropies = {"train": torch.zeros(args.epochs,5+args.sources)}
metrics = {"test": torch.zeros(args.epochs,2)}

print('Decoder unfreeze after 100 epochs: ' + str(args.unfreeze_decoders_after_100))

for epoch in range(1, args.epochs+1):
    beta = 1.0
    
    if args.unfreeze_decoders_after_100:
        if epoch == 100:
            print("Before unfreeze: ", optimizer_all.param_groups[0]['params'][0].requires_grad)
            learning_rate_e_100 = optimizer_all.param_groups[0]['lr']
            model.unfreeze_decoder()
            optimizer_all = torch.optim.Adam(model.parameters(), lr = learning_rate_e_100)
            print('Decoder weights will be now updated', flush=True)
            print("After unfreeze: ", optimizer_all.param_groups[0]['params'][0].requires_grad)
            print("Decoder weight update status: ", model.Decoders[0][0].block[0].weight.requires_grad)
    
    if args.unfreeze_decoders:
        learning_rate = optimizer_all.param_groups[0]['lr']
        model.unfreeze_decoder()
        optimizer_all = torch.optim.Adam(model.parameters(), 
                                         lr = learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer_all, 
                                              step_size=1, 
                                              gamma=args.decay, 
                                              last_epoch=-1)
        print('Decoders weights will be now updated', flush=True)

    losses["train"][epoch-1], pi_list, b, entropies["train"][epoch-1] = train(epoch, b)
    losses["test"][epoch-1], metrics["test"][epoch-1] = test(epoch, pi_list, b)
    
    to_log = {"lr": torch.tensor(optimizer_all.param_groups[0]['lr'])}
    if to_log is not None:
        logger.append_and_write(**to_log)

    with torch.no_grad():
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), 
                       args.save_path + 'model_'+('vae' if args.variational else 'ae')+'_K' + str(args.sources) +  '.pt')

plot_losses(losses)
plot_entropy_decoder(entropies)
plot_entropy_encoders(entropies)
plot_entropy_priors(entropies)
plot_entropy_s_posterior(entropies)
plot_entropy_sum_with_ELBO(losses, entropies)
plot_accuracy(metrics)
