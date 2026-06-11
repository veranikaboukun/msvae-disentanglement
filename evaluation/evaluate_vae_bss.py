# Copyright (C) 2021 Julian Neri, Roland Badeau, Philippe Depalle
# Copyright (C) 2026 Veranika Boukun veranika.boukun@uni-oldenburg.de
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
# - Modified evaluation procedures for better MS-VAE compatibility 
# for ECML PKDD 2026 contribution "Disentanglement in a Multi-Stream VAE"
# - The results presented in Table 1 were obtained by using this code
# Original repository: https://github.com/jundsp/VAE-BSS/tree/main
# ----------------------------------------------------------------------

import torch
from torchvision import datasets, transforms
import numpy as np
import argparse
from src.model import *
from src.dataloader import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

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
            print(f"No samples with num_active_sources == {num_active_sources}")
            return 0.0, 0.0
        
        # Filter the data
        gt_images = [gt_images[i] for i in valid_indices]
        pred_images = [pred_images[i] for i in valid_indices]
        if labels is not None:
            labels = [labels[i] for i in valid_indices]
    
    for i in range(len(gt_images)):
        gt = gt_images[i].detach().cpu().numpy()
        pred = pred_images[i].detach().cpu().numpy()
        
        gt = gt / gt.max() if gt.max() > 1.0 else gt
        pred = pred / pred.max() if pred.max() > 1.0 else pred
        
        ssim_matrix = np.zeros((len(gt), len(pred)))
        for p in range(len(pred)):
            for g in range(len(gt)):
                ssim_matrix[p, g] = structural_similarity(gt[g], pred[p], data_range=1.0)
        
        row_ind, col_ind = linear_sum_assignment(-ssim_matrix)
        
        # Compute metrics
        for p, g in zip(row_ind, col_ind):
            psnr = peak_signal_noise_ratio(gt[g], pred[p], data_range=1.0)
            ssim = ssim_matrix[p, g]
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

    return psnr_scores, ssim_scores

def make_grid_panel_torch(image_list, grid_rows, grid_cols, sep=1, save_path=None, title=None):
    
    H, W = image_list[0].shape
    panel_h = grid_rows * H + (grid_rows - 1) * sep
    panel_w = grid_cols * W + (grid_cols - 1) * sep
    panel = torch.ones((panel_h, panel_w), dtype=image_list[0].dtype) 
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
    plt.imshow(panel_np, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def make_double_grid_panel_torch(gt_images, grid_rows, grid_cols, gap=10, save_path=None, title=None):
    
    N = gt_images[0].shape[0]
    H, W = gt_images[0].shape[1:]
   
    sources = [[gt_images[i][s] for i in range(len(gt_images))] for s in range(N)]
    
    # Create a grid for each source
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, 
                        default='pretrained_model_files/model_vae_K2_vae_bss.pt')
    
    parser.add_argument('--data-directory', type=str, 
                        default='/data/mnist_test_set_K_2_noiseless.pt', metavar='fname',
                        help='MNIST dataset')
    
    parser.add_argument('--num-sources', type=int, default=2,
                        help='Number of sources to infer')
    
    parser.add_argument('--testing-indices-0', type=str, 
                        default='data/indices_testing-zeros.npy', metavar='fname',
                        help='Digit 0 data not used for pretesting') 
    
    parser.add_argument('--testing-indices-1', type=str, 
                        default='data/indices_testing-ones.npy', metavar='fname',
                        help='Digit 1 data not used for pretesting') 
    
    parser.add_argument('--testing-indices-2', type=str, 
                        default='data/indices_testing-twos.npy', metavar='fname',
                        help='Digit 2 data not used for pretesting')
    
    parser.add_argument('--testing-indices-3', type=str, 
                        default='data/indices_testing-threes.npy', metavar='fname',
                        help='Digit 3 data not used for pretesting')
    
    parser.add_argument('--testing-indices-4', type=str, 
                        default='data/indices_testing-fours.npy', metavar='fname',
                        help='Digit 4 data not used for pretesting')
    
    parser.add_argument('--testing-indices-5', type=str, 
                        default='data/indices_testing-fives.npy', metavar='fname',
                        help='Digit 5 data not used for pretesting')
    
    parser.add_argument('--testing-indices-6', type=str, 
                        default='data/indices_testing-sixes.npy', metavar='fname',
                        help='Digit 6 data not used for pretesting')
    
    parser.add_argument('--testing-indices-7', type=str, 
                        default='data/indices_testing-sevens.npy', metavar='fname',
                        help='Digit 7 data not used for pretesting')
    
    parser.add_argument('--testing-indices-8', type=str, 
                        default='data/indices_testing-eights.npy', metavar='fname',
                        help='Digits 8 data not used for pretesting')
    
    parser.add_argument('--testing-indices-9', type=str, 
                        default='data/indices_testing-nines.npy', metavar='fname',
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
                    help='Generative pi for 0.')
    parser.add_argument('--pi-1-gen', type=float, default=0.1,
                        help='Generative pi for 1.')
    parser.add_argument('--pi-2-gen', type=float, default=0.1,
                        help='Generative pi for 2.')
    parser.add_argument('--pi-3-gen', type=float, default=0.1,
                        help='Generative pi for 3.')
    parser.add_argument('--pi-4-gen', type=float, default=0.1,
                        help='Generative pi for 4.')
    parser.add_argument('--pi-5-gen', type=float, default=0.1,
                        help='Generative pi for 5.')
    parser.add_argument('--pi-6-gen', type=float, default=0.1,
                        help='Generative pi for 6.')
    parser.add_argument('--pi-7-gen', type=float, default=0.1,
                        help='Generative pi for 7.')
    parser.add_argument('--pi-8-gen', type=float, default=0.1,
                        help='Generative pi for 8.')
    parser.add_argument('--pi-9-gen', type=float, default=0.1,
                        help='Generative pi for 9.')
    
    parser.add_argument('--scale', type=float, default=1e-6,
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

    # (!) Use code below to generate your own dataset:
    # mnist_test_set = mix_mnist_data_return_individual_sources(valid_subset_list, 
    #                                                             num_samples_test, 
    #                                                             pi_gen_values, 
    #                                                             b_gen)

    # Read in the provided test file
    data = torch.load(args.path_dataset)
    mixtures = data["mixtures"]
    sources = data["sources"]
    individual_images = data["individual_images"]

    mnist_test_set = TensorDataset(mixtures, sources, individual_images)

    mnist_dataloader_test = DataLoader(mnist_test_set, 
                                       batch_size=args.batch_size, 
                                       shuffle=False)
   
    metrics = torch.zeros(3)
    print('MNIST digit count = ' + str(args.num_sources))
    print('\tK = ' + str(args.num_sources))

    # Load the pretrained VAE-BSS Model
    model_vae = VAE(dimx=args.dimx,
                    dimz=args.dimz,
                    n_sources=args.num_sources,
                    device=device,
                    variational=True).to(device)
    model_vae.load_state_dict(torch.load(args.model_path))
    
    model_vae.eval()


    with torch.no_grad():
        for i, (data, labels, individual_sources) in enumerate(mnist_dataloader_test):
            data = data.to(device).view(-1, args.dimx)
        
            sources_mask = labels.squeeze(-1) 
            num_active_sources = sources_mask.sum(dim=1)
            selected_indices_k = (num_active_sources == args.num_active_sources)

            selected_sources_mask_k = sources_mask[selected_indices_k]
            selected_individual_sources_k = individual_sources[selected_indices_k] 

            selected_data_k = data[selected_indices_k]

            activated_images_selected_k = []
            for i in range(selected_individual_sources_k.shape[0]):
                mask = selected_sources_mask_k[i] > 0
                images = selected_individual_sources_k[i][mask]
                images = images.squeeze(1) 
                activated_images_selected_k.append(images)
            

            activated_images_selected_k_batch = torch.stack(activated_images_selected_k, dim=0)
            
            # VAE Evaluation
            start_time_counter = time.time()
            x_vae, mu_z, logvar_z, s_vae = model_vae(selected_data_k) 
            end_time_counter = time.time()
            inference_time = end_time_counter - start_time_counter
            metrics[0] += round(inference_time, 2)

            x_vae = x_vae.view(-1,1,28,28)

            active_per_batch = x_vae.size(0)
            s_vae_selected = s_vae.reshape(active_per_batch, args.num_active_sources, 28, 28) 

            # (!) VAE-BSS no mask
            activated_preds_selected_k = [s_vae_selected[i] for i in range(s_vae_selected.shape[0])] 

            # (!) Please keep uncommented below to use VAEM-BSS
            activated_images_selected_k_nested = [t.tolist() for t in activated_images_selected_k]
            s_tru = torch.tensor(activated_images_selected_k_nested)  
            s_vae_selected_permute = optimal_permute(s_vae_selected.detach().cpu(), s_tru)
            data_selected = data[selected_indices_k]
            data_selected = data_selected.view(-1, 28, 28).detach().cpu()
            s_vaem = vae_masks(s_vae_selected_permute, data_selected)
            activated_preds_selected_k = [s_vaem[i] for i in range(s_vaem.shape[0])] 

            psnr_scores_k = []
            ssim_scores_k = []

            psnr_scores_k, ssim_scores_k = compute_permutation_invariant_metrics(gt_images=activated_images_selected_k,
                                                                pred_images=activated_preds_selected_k, 
                                                                num_active_sources=args.num_active_sources)

            metrics[1] += round(np.mean(psnr_scores_k), 2)
            metrics[2] += round(np.mean(ssim_scores_k), 2)


        num_examples = 9
        grid_size = 3

        # (!) Uncomment the section below to produce plots:
        # mixture_imgs = [data[i].cpu().detach().view(28,28) for i in range(num_examples)]
       
        # make_grid_panel_torch(mixture_imgs, 
        #                       grid_size, 
        #                       grid_size, 
        #                       save_path="mixture_panel.pdf", 
        #                       title="Mixtures")
        
        # make_double_grid_panel_torch(activated_preds_selected_k, 
        #                              grid_size, 
        #                              grid_size, 
        #                              gap=18, 
        #                              save_path="VAEM-BSS_panel-K-.pdf", 
        #                              title="VAEM-BSS")

        
        metrics /= len(mnist_dataloader_test)
        print('====> Inference time: {:.4f}'.format(metrics[0]), flush=True)
        print('====> Mean PSNR score: {:.4f}'.format(metrics[1]), flush=True)
        print('====> Mean SSIM score: {:.4f}'.format(metrics[2]), flush=True)

    print('Evaluation complete.')