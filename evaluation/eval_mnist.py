import numpy as np
import torch
import argparse
from distutils.util import strtobool
from src.model_streams import *
from src.dataloader import *
import itertools
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def run_inference(pi_list, b): 

    model.eval()
    metrics = torch.zeros(5) 

    with torch.no_grad():
        for i, (data, labels, individual_sources) in enumerate(mnist_dataloader_test):
           
            data = data.to(device).view(-1, args.dimx)
            data = data.float()

            mu_theta_list, mu_phi_list, logvar_phi_list, z_list = model(data, M=args.m_samples)

            _, _, _, _, _, _, q_s_x = loss_overall(data, mu_theta_list, 
                                                    mu_phi_list, logvar_phi_list, 
                                                    s_list, pi_list, z_list, 
                                                    beta=1, N=args.batch_size, 
                                                    M=args.m_samples, scale=b_test)

    
            # now computing the accuracy correctly
            labels_s_list = s_list.detach().cpu() @ labels.T.long()
            _, target = torch.max(labels_s_list, dim=1) 
            _, prediction = torch.max(q_s_x.detach().cpu(), dim=1)

            accuracy = accuracy_score(target.squeeze(0), prediction)

            metrics[0] += round(accuracy, 2)

            sources_mask = labels.squeeze(-1) 
            num_active_sources = sources_mask.sum(dim=1)  
            selected_indices_k2 = (num_active_sources == 2)
            selected_indices_k3 = (num_active_sources == 3)

            selected_sources_mask_k2 = sources_mask[selected_indices_k2]
            selected_individual_sources_k2 = individual_sources[selected_indices_k2] 

            selected_sources_mask_k3 = sources_mask[selected_indices_k3]
            selected_individual_sources_k3 = individual_sources[selected_indices_k3]  

            activated_images_selected_k2 = []
            for i in range(selected_individual_sources_k2.shape[0]):
                mask = selected_sources_mask_k2[i] > 0
                images = selected_individual_sources_k2[i][mask]
                images = images.squeeze(1) 
                activated_images_selected_k2.append(images)
            
            activated_images_selected_k3 = []
            for i in range(selected_individual_sources_k3.shape[0]):
                mask = selected_sources_mask_k3[i] > 0
                images = selected_individual_sources_k3[i][mask]
                images = images.squeeze(1) 
                activated_images_selected_k3.append(images)

            # predictions
            mu_theta_tensor = torch.stack(mu_theta_list, dim=0)
            # Rearrange to shape: (num_samples, num_sources, 28*28)
            mu_theta_tensor = mu_theta_tensor.permute(1, 0, 2, 3).squeeze(3)

            selected_mu_theta_k2 = mu_theta_tensor[selected_indices_k2]   
            activated_preds_selected_k2 = []

            for i in range(selected_mu_theta_k2.shape[0]):      
                source_mask_k2 = selected_sources_mask_k2[i] > 0   
                preds_k2 = selected_mu_theta_k2[i][source_mask_k2]   
                preds_images_k2 = preds_k2.view(-1, 28, 28)       
                activated_preds_selected_k2.append(preds_images_k2)

            
            selected_mu_theta_k3 = mu_theta_tensor[selected_indices_k3]   
            activated_preds_selected_k3 = []

            for i in range(selected_mu_theta_k3.shape[0]):      
                source_mask_k3 = selected_sources_mask_k3[i] > 0   
                preds_k3 = selected_mu_theta_k3[i][source_mask_k3] 
                preds_images_k3 = preds_k3.view(-1, 28, 28)      
                activated_preds_selected_k3.append(preds_images_k3)
            
            psnr_scores_k2 = []
            ssim_scores_k2 = []

            for gt_images, pred_images in zip(activated_images_selected_k2, activated_preds_selected_k2):
                # Both gt_images and pred_images: (num_active_sources, 28, 28)
                for gt, pred in zip(gt_images, pred_images):
                    # Convert to numpy, ensure no gradients etc
                    gt_np = gt.detach().cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()
                    # Compute metrics
                    psnr = peak_signal_noise_ratio(gt_np, pred_np)
                    ssim = structural_similarity(gt_np, pred_np, data_range=gt_np.max() - gt_np.min())
                    psnr_scores_k2.append(psnr)
                    ssim_scores_k2.append(ssim)

            metrics[1] += round(np.mean(psnr_scores_k2), 2)
            metrics[2] += round(np.mean(ssim_scores_k2), 2)

            psnr_scores_k3 = []
            ssim_scores_k3 = []

            for gt_images, pred_images in zip(activated_images_selected_k3, activated_preds_selected_k3):
                # Both gt_images and pred_images: (num_active_sources, 28, 28)
                for gt, pred in zip(gt_images, pred_images):
                    # Convert to numpy, ensure no gradients etc
                    gt_np = gt.detach().cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()
                    # Compute metrics
                    psnr = peak_signal_noise_ratio(gt_np, pred_np)
                    ssim = structural_similarity(gt_np, pred_np, data_range=gt_np.max() - gt_np.min())
                    psnr_scores_k3.append(psnr)
                    ssim_scores_k3.append(ssim)

            metrics[3] += round(np.mean(psnr_scores_k3), 2)
            metrics[4] += round(np.mean(ssim_scores_k3), 2)


        metrics /= len(mnist_dataloader_test)
        print('====> Test accuracy: {:.4f}'.format(metrics[0]), flush=True)
        print('====> Mean PSNR score (K = 2): {:.4f}'.format(metrics[1]), flush=True)
        print('====> Mean SSIM score (K = 2): {:.4f}'.format(metrics[2]), flush=True)
        print('====> Mean PSNR score (K = 3): {:.4f}'.format(metrics[3]), flush=True)
        print('====> Mean SSIM score (K = 3): {:.4f}'.format(metrics[4]), flush=True)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-path', type=str, 
                        default='/training/MNIST/MS-VAE-model.pt')
    
    parser.add_argument('--data-directory', type=str, 
                        default='/data/MNIST/', metavar='fname',
                        help='Folder MNIST dataset')
    
    parser.add_argument('--testing-indices-0', type=str, 
                        default='/indices_testing-zeros.npy', metavar='fname',
                        help='Digit 0 data not used for pretesting') 
    
    parser.add_argument('--testing-indices-1', type=str, 
                        default='/indices_testing-ones.npy', metavar='fname',
                        help='Digit 1 data not used for pretesting') 
    
    parser.add_argument('--testing-indices-2', type=str, 
                        default='/indices_testing-twos.npy', metavar='fname',
                        help='Digit 2 data not used for pretesting')
    
    parser.add_argument('--testing-indices-3', type=str, 
                        default='/indices_testing-threes.npy', metavar='fname',
                        help='Digit 3 data not used for pretesting')
    
    parser.add_argument('--testing-indices-4', type=str, 
                        default='/indices_testing-fours.npy', metavar='fname',
                        help='Digit 4 data not used for pretesting')
    
    parser.add_argument('--testing-indices-5', type=str, 
                        default='/indices_testing-fives.npy', metavar='fname',
                        help='Digit 5 data not used for pretesting')
    
    parser.add_argument('--testing-indices-6', type=str, 
                        default='/indices_testing-sixes.npy', metavar='fname',
                        help='Digit 6 data not used for pretesting')
    
    parser.add_argument('--testing-indices-7', type=str, 
                        default='/indices_testing-sevens.npy', metavar='fname',
                        help='Digit 7 data not used for pretesting')
    
    parser.add_argument('--testing-indices-8', type=str, 
                        default='/indices_testing-eights.npy', metavar='fname',
                        help='Digits 8 data not used for pretesting')
    
    parser.add_argument('--testing-indices-9', type=str, 
                        default='/indices_testing-nines.npy', metavar='fname',
                        help='Digit 9 data not used for pretesting')
    
    parser.add_argument('--variational', type=str, default=True,
                    help='VAE if True, AE if False.')
    
    parser.add_argument('--sources', type=int, default=10,
                    help='Number of sources to infer')
    
    parser.add_argument('--dimz', type=int, default=20,
                    help='Dimension of latent space.')
    
    parser.add_argument('--dimx', type=int, default=784,
                    help='Dimension of latent space.')
    
    parser.add_argument('--m_samples', type=int, default=1,
                        help='Number of M samples used to approximate the integrals')
    
    parser.add_argument('--prior', type=str, default='gauss',
                    help='Prior over latent variables')
    
    parser.add_argument('--batch-size', type=int, default=125, metavar='N',
                    help='batch size')
    
    parser.add_argument('--n-test', type=int, default=5000,
                    help='Number of testing samples to generate.')
    
    # generative parameters
    parser.add_argument('--pi-0-gen', type=float, default=0.1,
                    help='Generative pi for signal 1.')
    parser.add_argument('--pi-1-gen', type=float, default=0.1,
                        help='Generative pi for signal 1.')
    parser.add_argument('--pi-2-gen', type=float, default=0.1,
                        help='Generative pi for signal 2.')
    parser.add_argument('--pi-3-gen', type=float, default=0.1,
                        help='Generative pi for signal 1.')
    parser.add_argument('--pi-4-gen', type=float, default=0.1,
                        help='Generative pi for signal 2.')
    parser.add_argument('--pi-5-gen', type=float, default=0.1,
                        help='Generative pi for signal 1.')
    parser.add_argument('--pi-6-gen', type=float, default=0.1,
                        help='Generative pi for signal 2.')
    parser.add_argument('--pi-7-gen', type=float, default=0.1,
                        help='Generative pi for signal 1.')
    parser.add_argument('--pi-8-gen', type=float, default=0.1,
                        help='Generative pi for signal 2.')
    parser.add_argument('--pi-9-gen', type=float, default=0.1,
                        help='Generative pi for signal 1.')

    # learned parameters
    parser.add_argument('--pi-0', type=float, default=0.10374,
                        help='learned pi for signal 0.')
    parser.add_argument('--pi-1', type=float, default=0.10392,
                        help='learned pi for signal 1.')
    parser.add_argument('--pi-2', type=float, default=0.09648,
                        help='learned pi for signal 2.')
    parser.add_argument('--pi-3', type=float, default=0.10272,
                        help='learned pi for signal 3.')
    parser.add_argument('--pi-4', type=float, default=0.0994,
                        help='learned pi for signal 4.')
    parser.add_argument('--pi-5', type=float, default=0.09098,
                        help='learned pi for signal 5.')
    parser.add_argument('--pi-6', type=float, default=0.10066,
                        help='learned pi for signal 6.')
    parser.add_argument('--pi-7', type=float, default=0.09884,
                        help='learned pi for signal 7.')
    parser.add_argument('--pi-8', type=float, default=0.09764,
                        help='learned pi for signal 8.')
    parser.add_argument('--pi-9', type=float, default=0.09964,
                        help='learned pi for signal 9.')
    parser.add_argument('--b', type=float, default=0.11289,
                        help='learned Scale of Laplace likelihood')
    
    parser.add_argument('--scale', type=float, default=0.1,
                    help='Scale of Laplace likelihood')
    
    parser.add_argument('--save-path', type=str, 
                        default='/evaluation/')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    
    parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of data loading workers (parallel).')
    
    parser.add_argument('--freeze-decoder', default=True, type=lambda x: bool(strtobool(x)), 
                    help='True if the encoder weight should be frozen (not updated).')
    
    parser.add_argument('--freeze-encoder', default=True, type=lambda x: bool(strtobool(x)), 
                    help='True if the decoder weight should be frozen (not updated).')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

    test_set = datasets.MNIST(args.data_directory, train=False, download=True, transform=transforms.ToTensor())

    valid_subset_list = []
    pi_gen_values = []

    testing_indices = {}

    for i in range(args.sources):

        testing_key = f'testing_indices_{i}'
        
        testing_indices[i] = np.load(getattr(args, testing_key))
        npy_testing_indices = testing_indices[i]
        
        # Process the testing/validation data
        valid_with_label = Subset(test_set, npy_testing_indices)
        valid_subset = ImageOnlyDataset(valid_with_label)
        valid_subset_list.append(valid_subset)  # Store in list

        pi_gen_key = f'pi_{i}_gen'
        pi_gen_value = getattr(args, pi_gen_key)
        pi_gen_values.append(pi_gen_value)
    
    # Construct the string using join for a clean print statement
    pi_gen_str = ", ".join(str(pi) for pi in pi_gen_values)
    
    # Print the result
    print("Generative Bernoulli (pi) parameters: " + pi_gen_str)

    num_samples_test = args.n_test #1000

    print('Number of generated testing datapoints: ' + str(num_samples_test))

    b_gen = args.scale
    print("Generative noise scale (b): " + str(b_gen))

    mnist_test_set = mix_mnist_data_return_individual_sources(valid_subset_list, 
                                                                num_samples_test, 
                                                                pi_gen_values, b_gen)
    
    mnist_dataloader_test = DataLoader(mnist_test_set, batch_size=args.batch_size, shuffle=True)

    H = args.sources
    s_list = torch.asarray(list(itertools.product(range(2), repeat=H))).to(device)

    b_test = args.b
    print("Learned Laplace scale (b): " + str(b_test))

    pi_list = []

    for i in range(args.sources):
        pi_key = f'pi_{i}'
        pi_value = getattr(args, pi_key)
        pi_list.append(pi_value)

    pi_learned_str = ", ".join(str(pi) for pi in pi_list)
    print("Learned Bernoulli parameters pi: " + pi_learned_str)

    model = MultiStream_VAE(dimx=args.dimx, dimz=args.dimz, n_sources=args.sources, 
                        device=device, variational=args.variational, 
                        freeze_decoder=args.freeze_decoder, freeze_encoder=args.freeze_encoder).to(device)
    
    
    model.load_state_dict(torch.load(args.model_path))  

    loss_overall = Loss_k_sources(sources=args.sources, likelihood='laplace', variational=args.variational, prior=args.prior)

    start_time_counter = time.time()
    metrics = run_inference(pi_list, b_test)
    # End timer
    end_time_counter = time.time()

    # Calculate elapsed time
    inference_time = end_time_counter - start_time_counter
    print(f"Inference Time (Metrics): {inference_time:.4f} seconds")

    print('_')