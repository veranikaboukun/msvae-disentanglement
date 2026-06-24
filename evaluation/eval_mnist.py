# Copyright (C) 2026 Veranika Boukun (veranika.boukun@uni-oldenburg.de), Jörg Lücke

import numpy as np
import torch
import argparse
from distutils.util import strtobool
from src.model_streams import *
from src.dataloader import *
import itertools
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def compute_permutation_invariant_metrics(gt_images, 
                                        pred_images, 
                                        labels=None, 
                                        num_active_sources=None):

    psnr_scores = []
    ssim_scores = []
    
    if num_active_sources is not None:
        if isinstance(num_active_sources, int):
            valid_indices = list(range(len(gt_images)))
        else:
            valid_indices = [i for i, n in enumerate(num_active_sources) if n == num_active_sources]
        
        if not valid_indices:
            return 0.0, 0.0
        
        gt_images = [gt_images[i] for i in valid_indices]
        pred_images = [pred_images[i] for i in valid_indices]
        if labels is not None:
            labels = [labels[i] for i in valid_indices]
    
    for i in range(len(gt_images)):
        gt = gt_images[i].detach().cpu().numpy()
        pred = pred_images[i].detach().cpu().numpy()
        
        gt = gt / gt.max() if gt.max() > 1.0 else gt
        pred = pred / pred.max() if pred.max() > 1.0 else pred
        
        row_ind = torch.arange(num_active_sources)
        col_ind = torch.arange(num_active_sources)
        
        # Compute metrics
        for p, g in zip(row_ind, col_ind):
            if p < len(pred):
                pred_img = pred[p]
            else:
                lap_dist = Laplace(torch.zeros_like(torch.from_numpy(gt[g])), args.scale)
                pred_img = lap_dist.sample().cpu().numpy()
            if gt[g].shape == pred_img.shape:
                psnr = peak_signal_noise_ratio(gt[g], pred_img, data_range=1.0)
                ssim = structural_similarity(gt[g], pred_img, data_range=1.0)
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)
            else:
                pass  

    return psnr_scores, ssim_scores

def make_double_grid_panel_torch(gt_images, grid_rows, grid_cols, gap=10, save_path=None, title=None):

    N = gt_images[0].shape[0]
    H, W = gt_images[0].shape[1:]
    
    sources = [[gt_images[i][s] for i in range(len(gt_images))] for s in range(N)]  # sources[s][i]
   
    grids = []
    for s in range(N):
        grid = torch.ones((grid_rows*H + (grid_rows-1)*2, grid_cols*W + (grid_cols-1)*2))
        idx = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                if idx < len(gt_images):
                    y = r * (H + 2)
                    x = c * (W + 2)
                    grid[y:y+H, x:x+W] = sources[s][idx]
                    idx += 1
        grids.append(grid)
    
    big_gap = torch.ones((gap, grids[0].shape[1]))
    panel = grids[0]
    for g in grids[1:]:
        panel = torch.cat([panel, big_gap, g], dim=0)

    panel_np = panel.cpu().numpy()
    plt.figure(figsize=(grid_cols*2, grid_rows*2*N + 2))
    plt.imshow(panel_np, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def make_grid_panel_torch(image_list, grid_rows, grid_cols, sep=1, save_path=None, title=None):

    H, W = image_list[0].shape
    max_value = max(img.max().item() for img in image_list)  
    min_value = min(img.min().item() for img in image_list)  
    panel_h = grid_rows * H + (grid_rows - 1) * sep
    panel_w = grid_cols * W + (grid_cols - 1) * sep
    panel = torch.full((panel_h, panel_w), max_value, dtype=image_list[0].dtype) 
    idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            if idx < len(image_list):
                y = r * (H + sep)
                x = c * (W + sep)
                panel[y:y+H, x:x+W] = image_list[idx]
                idx += 1
    panel_np = panel.cpu().numpy()
    plt.figure(figsize=(grid_cols*2, grid_rows*2))
    plt.imshow(panel_np, cmap='gray', vmin=min_value, vmax=max_value)
    plt.axis('off')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


def run_inference(): 

    model.eval()
    metrics = torch.zeros(8) #0: accuracy, 1: PSNR, 2: SSIM, 3:PSNR, 4: SSIM, 5: inference time

    with torch.no_grad():
        for i, (data, labels, individual_sources) in enumerate(mnist_dataloader_test):
           
            data = data.to(device).view(-1, args.dimx)
            data = data.float()

            start_time_counter = time.time()
            mu_theta_list, mu_phi_list, logvar_phi_list, z_list = model(data, 
                                                                        M=args.m_samples)
            end_time_counter = time.time()
            inference_time = end_time_counter - start_time_counter
            metrics[7] += round(inference_time, 2)

            # fixed K
            _, _, _, _, _, q_s_x = loss_overall(data, 
                                                mu_theta_list, 
                                                mu_phi_list, 
                                                logvar_phi_list, 
                                                s_list_K, 
                                                z_list, 
                                                K = args.num_active_sources,
                                                beta=1, 
                                                N=args.batch_size, 
                                                M=args.m_samples, 
                                                scale=b_test)

            # q_s_x shape fixed K
            labels_s_list = s_list_K.detach().cpu() @ labels.T.long() # (45, 125)
            _, target = torch.max(labels_s_list, dim=0) # correct with visual inspection
            _, prediction = torch.max(q_s_x.detach().cpu(), dim=1)

            accuracy = accuracy_score(target.squeeze(0), prediction)

            # same for all:
            metrics[0] += round(accuracy, 2)
            
            # select ground truth
            sources_mask = labels.squeeze(-1) 
            num_active_sources = sources_mask.sum(dim=1) 
            selected_indices_k = (num_active_sources == args.num_active_sources)

            selected_sources_mask_k = sources_mask[selected_indices_k]
            selected_individual_sources_k = individual_sources[selected_indices_k]  

            activated_images_selected_k = []
            for i in range(selected_individual_sources_k.shape[0]):
                mask = selected_sources_mask_k[i] > 0
                images = selected_individual_sources_k[i][mask]
                images = images.squeeze(1) 
                activated_images_selected_k.append(images)

            # Predictions of outputs for both fixed K and Bernoulli:
            _, max_indices = torch.max(q_s_x.detach().cpu(), dim=1)

            mu_theta_i_list = []
            for i in range(args.sources):
                mu_theta_i = mu_theta_list[i].sum(1).detach().cpu()
                mu_theta_i_list.append(mu_theta_i)

            mu_theta_i_new_list = [] 
            mu_theta_only_active_list = []  

            # For each source, build per-source new tensor (all predictions)
            for i, mu_theta_i in enumerate(mu_theta_i_list):
                mu_theta_i_new = torch.zeros_like(mu_theta_i).to(device)
                for n, index in enumerate(max_indices):
                    # fixed K
                    s_vector = s_list_K[index].detach().cpu().float() 
                    s_i = s_vector[i]
                    mu_theta_i_new[n, ...] = mu_theta_i[n, ...] * s_i
                mu_theta_i_new_list.append(mu_theta_i_new)

            # For each sample, collect only the active predictions across all sources 
            for n, index in enumerate(max_indices):
                # fixed K
                s_vector = s_list_K[index].detach().cpu().float()
                active_images = []
                for i, mu_theta_i in enumerate(mu_theta_i_list):
                    s_i = s_vector[i]
                    if s_i > 0:
                        active_images.append(mu_theta_i[n, ...].view(28, 28)) # append only active images
                mu_theta_only_active_list.append(torch.stack(active_images)) 

            activated_preds_selected_k = mu_theta_only_active_list
            
            psnr_scores_k = []
            ssim_scores_k = []

            psnr_scores_k, ssim_scores_k = compute_permutation_invariant_metrics(gt_images=activated_images_selected_k,
                                                                                   pred_images=activated_preds_selected_k, 
                                                                                   num_active_sources=args.num_active_sources)

            metrics[1] += round(np.mean(psnr_scores_k), 2)
            metrics[2] += round(np.mean(ssim_scores_k), 2)


        metrics /= len(mnist_dataloader_test)
        print('====> Test accuracy: {:.4f}'.format(metrics[0]), flush=True)
        print('====> Mean PSNR score: {:.4f}'.format(metrics[1]), flush=True)
        print('====> Mean SSIM score: {:.4f}'.format(metrics[2]), flush=True)
        print('====> Mean Inference time: {:.4f}'.format(metrics[7]), flush=True)

        ## 
        num_examples = 9
        grid_size = 3

        # mixture_imgs = [data[i].cpu().detach().view(28,28) for i in range(num_examples)]
       
        # make_grid_panel_torch(mixture_imgs, 
        #                       grid_size, 
        #                       grid_size, 
        #                       save_path="", 
        #                       title="Mixtures")

        # make_double_grid_panel_torch(activated_images_selected_k, 
        #                              grid_size, 
        #                              grid_size, 
        #                              gap=18, 
        #                              #n_sources=2,
        #                              save_path="", 
        #                              title="Original")
        
        # make_double_grid_panel_torch(activated_preds_selected_k, 
        #                              grid_size, 
        #                              grid_size, 
        #                              gap=10,  #K=2:18, K=3,4: 10
        #                              #n_sources=2,
        #                              save_path="", 
        #                              title="MS-VAE 10%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-path', type=str, 
                        default='msvae-disentanglement/evaluation/ms_vae_data_checkpoints/checkpoints/ms-vae/MS-VAE_10_model_checkpoint_K10.pt')
    
    parser.add_argument('--data-directory', type=str, 
                        default='msvae-disentanglement/evaluation/ms_vae_data_checkpoints/data/mnist_test_set_K_4_noiseless.pt', metavar='fname',
                        help='Folder MNIST dataset')
    
    parser.add_argument('--num-active-sources', type=int, default=4,
                    help='Number of sources that are active')
    
    parser.add_argument('--testing-indices-0', type=str, 
                        default='msvae-disentanglement/evaluation/index_files/indices_testing-zeros.npy', metavar='fname',
                        help='Digit 0 data not used for pretesting') 
    
    parser.add_argument('--testing-indices-1', type=str, 
                        default='msvae-disentanglement/evaluation/index_files/indices_testing-ones.npy', metavar='fname',
                        help='Digit 1 data not used for pretesting') 
    
    parser.add_argument('--testing-indices-2', type=str, 
                        default='msvae-disentanglement/evaluation/index_files/indices_testing-twos.npy', metavar='fname',
                        help='Digit 2 data not used for pretesting')
    
    parser.add_argument('--testing-indices-3', type=str, 
                        default='msvae-disentanglement/evaluation/index_files/indices_testing-threes.npy', metavar='fname',
                        help='Digit 3 data not used for pretesting')
    
    parser.add_argument('--testing-indices-4', type=str, 
                        default='msvae-disentanglement/evaluation/index_files/indices_testing-fours.npy', metavar='fname',
                        help='Digit 4 data not used for pretesting')
    
    parser.add_argument('--testing-indices-5', type=str, 
                        default='msvae-disentanglement/evaluation/index_files/indices_testing-fives.npy', metavar='fname',
                        help='Digit 5 data not used for pretesting')
    
    parser.add_argument('--testing-indices-6', type=str, 
                        default='msvae-disentanglement/evaluation/index_files/indices_testing-sixes.npy', metavar='fname',
                        help='Digit 6 data not used for pretesting')
    
    parser.add_argument('--testing-indices-7', type=str, 
                        default='msvae-disentanglement/evaluation/index_files/indices_testing-sevens.npy', metavar='fname',
                        help='Digit 7 data not used for pretesting')
    
    parser.add_argument('--testing-indices-8', type=str, 
                        default='msvae-disentanglement/evaluation/index_files/indices_testing-eights.npy', metavar='fname',
                        help='Digits 8 data not used for pretesting')
    
    parser.add_argument('--testing-indices-9', type=str, 
                        default='msvae-disentanglement/evaluation/index_files/indices_testing-nines.npy', metavar='fname',
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

    ## Uncomment lines for creating a new dataset (use path to MNIST dataset):
    # test_set = datasets.MNIST(args.data_directory, 
    #                           train=False, 
    #                           download=True, 
    #                           transform=transforms.ToTensor())

    # valid_subset_list = []
    # pi_gen_values = []

    # testing_indices = {}

    # for i in range(args.sources):

    #     testing_key = f'testing_indices_{i}'
        
    #     testing_indices[i] = np.load(getattr(args, testing_key))
    #     npy_testing_indices = testing_indices[i]
        
    #     # Process the testing/validation data
    #     valid_with_label = Subset(test_set, npy_testing_indices)
    #     valid_subset = ImageOnlyDataset(valid_with_label)
    #     valid_subset_list.append(valid_subset)  # Store in list

    #     pi_gen_key = f'pi_{i}_gen'
    #     pi_gen_value = getattr(args, pi_gen_key)
    #     pi_gen_values.append(pi_gen_value)
    
    # # Construct the string using join for a clean print statement
    # pi_gen_str = ", ".join(str(pi) for pi in pi_gen_values)
    
    # # Print the result
    # print("Generative Bernoulli (pi) parameters: " + pi_gen_str)

    # num_samples_test = args.n_test #1000

    # print('Number of generated testing datapoints: ' + str(num_samples_test))

    # b_gen = args.scale
    # print("Generative noise scale (b): " + str(b_gen))
    
    # mnist_test_set = mix_mnist_data_fixed_sources(valid_subset_list, 
    #                                                 num_samples_test, 
    #                                                 args.num_active_sources,
    #                                                 args.sources,  
    #                                                 args.scale,
    #                                                 seed=1234)

    ## Load an existing dataset
    data = torch.load(args.data_directory)
    mixtures = data["mixtures"]
    sources = data["sources"]
    individual_images = data["individual_images"]

    mnist_test_set = TensorDataset(mixtures, sources, individual_images)

    mnist_dataloader_test = DataLoader(mnist_test_set, batch_size=args.batch_size, shuffle=True)

    H = args.sources
    s_list_full = torch.asarray(list(itertools.product(range(2), repeat=H))).to(device)
    
    #fixed prior
    active_counts = torch.sum(s_list_full, axis=1)
    mask = (active_counts == args.num_active_sources)
    s_list_K = s_list_full[mask, :]

    b_test = args.b
    print("Learned Laplace scale (b): " + str(b_test))

    ## Pi-parameters are not required for fixed K evaluation
    # pi_list = []

    # for i in range(args.sources):
    #     pi_key = f'pi_{i}'
    #     pi_value = getattr(args, pi_key)
    #     pi_list.append(pi_value)

    # pi_learned_str = ", ".join(str(pi) for pi in pi_list)
    # print("Learned Bernoulli parameters pi: " + pi_learned_str)

    model = MultiStream_VAE(dimx=args.dimx, dimz=args.dimz, n_sources=args.sources, 
                        device=device, variational=args.variational, 
                        freeze_decoder=args.freeze_decoder, freeze_encoder=args.freeze_encoder).to(device)
    
    
    model.load_state_dict(torch.load(args.model_path))  

    loss_overall = Loss_MNIST_fixed_K(sources=args.sources, 
                                      likelihood='laplace', 
                                      variational=args.variational, 
                                      prior=args.prior) 
    
    metrics = run_inference()

    print('_')