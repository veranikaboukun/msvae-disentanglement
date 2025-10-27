import torch
from torchvision import datasets, transforms
import numpy as np
import argparse
from src.model import *
from src.dataloader import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
from torch.utils.data import DataLoader, Subset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, 
                        default='/training/MNIST/MS-VAE-model.pt')
    
    parser.add_argument('--data-directory', type=str, 
                        default='/data/MNIST/', metavar='fname',
                        help='Folder MNIST dataset')
    
    parser.add_argument('--num-sources', type=int, default=2,
                        help='Number of sources to infer')
    
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
    
    parser.add_argument('--prior', type=str, default='gauss',
                    help='Prior over latent variables')
    
    parser.add_argument('--batch-size', type=int, default=125, metavar='N',
                    help='batch size')
    
    parser.add_argument('--n-test', type=int, default=5000,
                    help='Number of testing samples to generate.')
    
    # Generative parameters
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
    
    parser.add_argument('--scale', type=float, default=0.1,
                    help='Scale of Laplace likelihood')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    
    parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of data loading workers (parallel).')

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
        valid_subset_list.append(valid_subset) 

        pi_gen_key = f'pi_{i}_gen'
        pi_gen_value = getattr(args, pi_gen_key)
        pi_gen_values.append(pi_gen_value)

    num_samples_test = args.n_test
    b_gen = args.scale

    mnist_test_set = mix_mnist_data_return_individual_sources(valid_subset_list, 
                                                                num_samples_test, 
                                                                pi_gen_values, 
                                                                b_gen)
        
    mnist_dataloader_test = DataLoader(mnist_test_set, batch_size=args.batch_size, shuffle=True)
   
    metrics = torch.zeros(5)
    print('\tK = ' + str(args.num_sources))

    # Load the Trained VAE-BSS Model
    model_vae = VAE(dimx=args.dimx,dimz=args.dimz,n_sources=args.num_sources,device=device,variational=True).to(device)
    model_vae.load_state_dict(torch.load(args.model_path))
    
    model_vae.eval()
    start_time_counter = time.time()

    with torch.no_grad():
        for i, (data, labels, individual_sources) in enumerate(mnist_dataloader_test):
            data = data.to(device).view(-1, args.dimx)
        
            # VAE Evaluation
            x_vae, mu_z, logvar_z, s_vae = model_vae(data) 
            x_vae = x_vae.view(-1,1,28,28)

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

           
            s_vae = s_vae.reshape(args.batch_size, args.num_sources, 28, 28) 
            s_vae_selected = s_vae[selected_indices_k2] 

            # Uncomment for VAE - no mask:  
            # activated_preds_selected_k2 = [s_vae_selected[i] for i in range(s_vae_selected.shape[0])] 

            # VAEM 
            activated_images_selected_k2_nested = [t.tolist() for t in activated_images_selected_k2]
            s_tru = torch.tensor(activated_images_selected_k2_nested)  
            s_vae_selected_permute = optimal_permute(s_vae_selected.detach().cpu(), s_tru)
            data_selected = data[selected_indices_k2]
            data_selected = data_selected.view(-1, 28, 28).detach().cpu()
            s_vaem = vae_masks(s_vae_selected_permute, data_selected)
            activated_preds_selected_k2 = [s_vaem[i] for i in range(s_vaem.shape[0])] # mask!
            
            # Uncomment for K = 3 #########
            # K = 3 sources
            # s_vae = s_vae.reshape(args.batch_size, args.num_sources, 28, 28)
            # s_vae_selected = s_vae[selected_indices_k3] 
            ## VAE - no mask
            # activated_preds_selected_k3 = [s_vae_selected[i] for i in range(s_vae_selected.shape[0])]

            # VAEM for K= 3 ###
            # activated_images_selected_k3_nested = [t.tolist() for t in activated_images_selected_k3]
            # s_tru = torch.tensor(activated_images_selected_k3_nested)  
            # s_vae_selected_permute = optimal_permute(s_vae_selected.detach().cpu(), s_tru)
            # data_selected = data[selected_indices_k3]
            # data_selected = data_selected.view(-1, 28, 28).detach().cpu()
            # s_vaem = vae_masks(s_vae_selected_permute, data_selected)
            # activated_preds_selected_k3 = [s_vaem[i] for i in range(s_vaem.shape[0])] # mask!

            # for i in range(selected_mu_theta_k3.shape[0]):      # For each retained sample
            #     source_mask_k3 = selected_sources_mask_k3[i] > 0   # shape: (10,) bool
            #     preds_k3 = selected_mu_theta_k3[i][source_mask_k3]    # (num_active_sources, 784)
            #     preds_images_k3 = preds_k3.view(-1, 28, 28)        # (num_active_sources, 28, 28)
            #     activated_preds_selected_k3.append(preds_images_k3)
            ##############################

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

            # Uncomment for K = 3 #########
            # psnr_scores_k3 = []
            # ssim_scores_k3 = []

            # for gt_images, pred_images in zip(activated_images_selected_k3, activated_preds_selected_k3):
            #     # Both gt_images and pred_images: (num_active_sources, 28, 28)
            #     for gt, pred in zip(gt_images, pred_images):
            #         # Convert to numpy, ensure no gradients etc
            #         gt_np = gt.detach().cpu().numpy()
            #         pred_np = pred.detach().cpu().numpy()
            #         # Compute metrics
            #         psnr = peak_signal_noise_ratio(gt_np, pred_np)
            #         ssim = structural_similarity(gt_np, pred_np, data_range=gt_np.max() - gt_np.min())
            #         psnr_scores_k3.append(psnr)
            #         ssim_scores_k3.append(ssim)

            # metrics[3] += round(np.mean(psnr_scores_k3), 2)
            # metrics[4] += round(np.mean(ssim_scores_k3), 2)
            ###############################

        metrics /= len(mnist_dataloader_test)
        print('====> Test accuracy: {:.4f}'.format(metrics[0]), flush=True)
        print('====> Mean PSNR score (K = 2): {:.4f}'.format(metrics[1]), flush=True)
        print('====> Mean SSIM score (K = 2): {:.4f}'.format(metrics[2]), flush=True)
        # print('====> Mean PSNR score (K = 3): {:.4f}'.format(metrics[3]), flush=True)
        # print('====> Mean SSIM score (K = 3): {:.4f}'.format(metrics[4]), flush=True)

        end_time_counter = time.time()
        # Calculate elapsed time
        inference_time = end_time_counter - start_time_counter
        print(f"Inference Time (Metrics): {inference_time:.4f} seconds")

    print('Evaluation complete.')