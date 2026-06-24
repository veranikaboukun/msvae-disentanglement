# Copyright (C) 2020 Vivek Jayaram and John Thickstun
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
# - Modified evaluation procedures for better MS-VAE compatibility 
# for ECML PKDD 2026 contribution "Disentanglement in a Multi-Stream VAE"
# - The results presented in Table 1 were obtained by using this code
# - Excludes Glow evaluation
# Original repository: https://github.com/jthickstun/basis-separation
# ----------------------------------------------------------------------

import os

import numpy as np
import torch
import torch.nn as nn
from ncsn.models.cond_refinenet_dilated import CondRefineNetDilated # please install from original NCSN repo
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import TensorDataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.optimize import linear_sum_assignment

import warnings
warnings.filterwarnings("ignore")

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
    H, W = gt_images[0].shape[2:]
   
    sources = [[gt_images[i][s] for i in range(len(gt_images))] for s in range(N)]
    
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

def compute_permutation_invariant_metrics(gt_images, pred_images):
    
    psnr_scores = []
    ssim_scores = []
    
    for i in range(len(gt_images)):
        gt = gt_images[i].detach().cpu().numpy().squeeze(1) 
        pred = pred_images[i].detach().cpu().numpy().squeeze(1) 
        
        gt = gt / gt.max() if gt.max() > 1.0 else gt
        pred = pred / pred.max() if pred.max() > 1.0 else pred
        
        ssim_matrix = np.zeros((2, 2))
        for p in range(2):
            for g in range(2):
                ssim_matrix[p, g] = structural_similarity(gt[g], pred[p], data_range=1.0)
        
        row_ind, col_ind = linear_sum_assignment(-ssim_matrix) 
        
        for p, g in zip(row_ind, col_ind):
            psnr = peak_signal_noise_ratio(gt[g], pred[p], data_range=1.0)
            ssim = ssim_matrix[p, g]
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
    
    return psnr_scores, ssim_scores

def dummy_train_iterator(args):
    print("args.batch_size value:", args.batch_size)
    print("type(args.batch_size):", type(args.batch_size))
    shape = (args.batch_size, 28, 28, 3)
    for idx, dim in enumerate(shape):
        print(f"dim {idx}: value={dim} type={type(dim)}")
    x = np.random.rand(*shape).astype(np.float32)
    y = np.zeros((args.batch_size,), dtype=np.int32)
    return x, y

def pytorch_test_iterator(args, test_dataset):

    for mixtures, sources, individual_images in test_dataset:

        mixtures_np = mixtures.cpu().detach().numpy()
        mixtures_np = mixtures_np.transpose(0, 2, 3, 1)
        mixtures_3_ch = np.repeat(mixtures_np, 3, axis=-1) # for the dummy 3 channels (is set like this by authors!)
        ys = np.zeros((args.batch_size,), dtype=np.int32)
        
        yield mixtures_3_ch, ys 

def test_iterator(test_iter):
    mixtures, _, _ = next(test_iter)
    mixtures_np = mixtures.cpu().detach().numpy().transpose(0, 2, 3, 1)
    mixtures_3_ch = np.repeat(mixtures_np, 3, axis=-1)
    return mixtures_3_ch

def get_single_image(dataset, batch_size):
    rand_idx = np.random.randint(0, dataset.data.shape[0] - 1, batch_size)
    image = dataset.data[rand_idx].float().view(batch_size, 1, 28, 28) / 255.

    return image

class MnistRunner():
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def test(self, dataset):
        # Load the score network
        states = torch.load(os.path.join(self.args.path_to_checkpoint_ncsn), map_location=self.device)
        state_dict = states[0]

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        
        scorenet = CondRefineNetDilated(logit_transform=False,
                                        ngf=64,
                                        num_classes=10,
                                        image_size=28).to(self.device)
       
        scorenet.load_state_dict(new_state_dict)
        scorenet.eval()

        metrics = torch.zeros(3)

        for i, (data, labels, individual_sources) in enumerate(dataset):
            
            sources_mask = labels.squeeze(-1) 
            num_active_sources = sources_mask.sum(dim=1)   
            selected_indices_k = (num_active_sources == self.args.num_active_sources)
            selected_sources_mask_k = sources_mask[selected_indices_k]
            selected_individual_sources_k = individual_sources[selected_indices_k] 

            activated_images_selected_k = []
            for i in range(selected_individual_sources_k.shape[0]):
                mask = selected_sources_mask_k[i] > 0
                images = selected_individual_sources_k[i][mask]
                activated_images_selected_k.append(images)
            
            activated_images_selected_k_batch = torch.stack(activated_images_selected_k, dim=0)

            gt_images = []
            for i in range(self.args.num_active_sources):
                gt_images.append(activated_images_selected_k_batch[:, i, ...]) 

            mixed = data.to(self.device) 
            xs = [] 
            for _ in range(self.args.num_active_sources):
                xs.append(nn.Parameter(torch.Tensor(self.args.batch_size, 1, 28, 28).uniform_()).cuda())

            step_lr=0.00003

            # This is noise amounts: "The noise σ is geometrically annealed from σ1 = 1.0 to σL = 0.01 with L = 10." (from paper)
            sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                            0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
            n_steps_each = 100

            for idx, sigma in enumerate(sigmas):
                lambda_recon = 1./(sigma**2)
                labels = torch.ones(self.args.batch_size, device=xs[i].device, dtype=torch.long) * idx

                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                start_time_counter = time.time()
                for step in range(n_steps_each):
                    noises = []
                    for _ in range(self.args.num_active_sources):
                        noises.append(torch.randn_like(xs[0]) * np.sqrt(step_size * 2))

                    grads = []
                    for i in range(self.args.num_active_sources):
                        result = scorenet(xs[i], labels).detach()
                        grads.append(result)

                    recon_loss = (torch.norm(torch.flatten(sum(xs) - mixed)) ** 2)
                    recon_grads = torch.autograd.grad(recon_loss, xs)

                    for i in range(self.args.num_active_sources): 
                        xs[i] = xs[i] + (step_size * grads[i]) + (-step_size * lambda_recon * recon_grads[i].detach()) + noises[i]

                end_time_counter = time.time()
                inference_time = end_time_counter - start_time_counter

            for i in range(self.args.num_active_sources):
                xs[i] = torch.clamp(xs[i], 0, 1)

            x_to_write = []
            for i in range(self.args.num_active_sources):
                x_to_write.append(torch.Tensor(xs[i].detach().cpu()))

            import pdb; pdb.set_trace()
            metrics[0] += round(inference_time, 2)

            psnr_scores_k = []
            ssim_scores_k = []

            activated_images_selected_k = [activated_images_selected_k_batch[i] for i in range(activated_images_selected_k_batch.shape[0])]
            x_to_write_tensor = torch.stack(x_to_write, dim=1)
            activated_preds_selected_k = [x_to_write_tensor[i] for i in range(x_to_write_tensor.shape[0])]

            import pdb; pdb.set_trace()
            psnr_scores_k, ssim_scores_k = compute_permutation_invariant_metrics(gt_images=activated_images_selected_k, 
                                                                                pred_images=activated_preds_selected_k) 

            metrics[1] += round(np.mean(psnr_scores_k), 2)
            metrics[2] += round(np.mean(ssim_scores_k), 2)

        metrics /= len(dataset)
        print('====> Inference time: {:.4f}'.format(metrics[0]), flush=True)
        print('====> Mean PSNR score (K = ' + str(self.args.num_active_sources) + '): {:.4f}'.format(metrics[1]), flush=True)
        print('====> Mean SSIM score (K = ' + str(self.args.num_active_sources) + '): {:.4f}'.format(metrics[2]), flush=True)

        # (!) Uncomment the section below to produce plots:
        # num_examples = 9
        # grid_size = 3
        # mixture_imgs = [data[i].cpu().detach().view(28,28) for i in range(num_examples)]

        # make_grid_panel_torch(mixture_imgs, 
        #                       grid_size, 
        #                       grid_size, 
        #                       save_path="mixture_panel_K_.pdf", 
        #                       title="Mixtures")
        
        # make_double_grid_panel_torch(activated_preds_selected_k, 
        #                              grid_size, 
        #                              grid_size, 
        #                              gap=18, 
        #                              save_path="BASIS_NCSN_panel_K_.pdf", 
        #                              title="BASIS (NCSN)")
        

def main():
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    data = torch.load(args.path_dataset)

    mixtures = data["mixtures"]
    sources = data["sources"]
    individual_images = data["individual_images"]

    mnist_test_set = TensorDataset(mixtures, sources, individual_images)
        
    test_dataset = DataLoader(mnist_test_set, 
                            batch_size=args.batch_size, 
                            shuffle=False)

    args.model == 'ncsn'
    runner = eval(args.runner)(args, device)
    runner.test(test_dataset)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path-to-checkpoint-ncsn', type=str, 
                        default='msvae-disentanglement/evaluation/pretrained_model_files/checkpoint.pth') # NCSN checkpoint file
    parser.add_argument('--path-dataset', type=str, 
                        default='msvae-disentanglement/evaluation/ms_vae_data_checkpoints/data/BASIS_compatible_mnist_test_set_K_2_noiseless.pt')
    
    parser.add_argument('--num-sources', type=int, default=10,
                    help='Number of sources to infer for the model')
    
    parser.add_argument('--num-active-sources', type=int, default=2,
                    help='')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    
    parser.add_argument('--batch-size', type=int, default=125, metavar='N',
                    help='batch size')
    
    parser.add_argument('--model', type=str, default='ncsn', help='The model to use as a prior')

    parser.add_argument('--runner', type=str, default='MnistRunner', help='The model to use as a prior') 

    parser.add_argument('--problem', type=str, default='mnist', help='') 


    main()