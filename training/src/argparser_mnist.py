# Copyright (C) 2026 Veranika Boukun (veranika.boukun@uni-oldenburg.de), Jörg Lücke

import argparse
from distutils.util import strtobool


parser = argparse.ArgumentParser(description='MS-VAE')

parser.add_argument('--sources', 
                    type=int, 
                    default=10,
                    help='Number of sources to infer')

parser.add_argument('--batch-size', 
                    type=int, 
                    default=125, 
                    metavar='N',
                    help='input batch size for training (default: 128)') 

parser.add_argument('--no-cuda', 
                    action='store_true', 
                    default=False,
                    help='enables CUDA training')

parser.add_argument('--seed', 
                    type=int, 
                    default=1, 
                    metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--data-directory', 
                    type=str, 
                    default='MNIST/data/', 
                    metavar='fname',
                    help='Folder containing the data')

parser.add_argument('--save-path', 
                    type=str, 
                    default='/training-MNIST-saves/', 
                    metavar='fname',
                    help='Folder for saving') 

# Use for 2 sources scenario (define the subset of digits)
parser.add_argument('--subset-1', 
                    type=int, 
                    default=0,
                    help='MNIST index subset 1, here 0s')

parser.add_argument('--subset-2', 
                    type=int, 
                    default=1,
                    help='MNIST index subset 2, here 1s')

parser.add_argument('--training-indices-0', 
                    type=str, 
                    default='training/index_files/indices_training-zeros.npy', 
                    metavar='fname',
                    help='remaining 0s data not used for pretraining') 

parser.add_argument('--testing-indices-0', 
                    type=str, 
                    default='evaluation/index_files/indices_testing-zeros.npy', 
                    metavar='fname',
                    help='remaining 0s data not used for pretesting') 

parser.add_argument('--training-indices-1',
                    type=str, 
                    default='training/index_files/indices_training-ones.npy',
                    metavar='fname',
                    help='remaining 1s data not used for pretraining') 

parser.add_argument('--testing-indices-1', 
                    type=str, 
                    default='evaluation/index_files/indices_testing-ones.npy', 
                    metavar='fname',
                    help='remaining 1s data not used for pretesting') 

parser.add_argument('--training-indices-2', 
                    type=str, 
                    default='training/index_files/indices_training-twos.npy', 
                    metavar='fname',
                    help='remaining 2s data not used for pretraining') 

parser.add_argument('--testing-indices-2', 
                    type=str, 
                    default='evaluation/index_files/indices_testing-twos.npy', 
                    metavar='fname',
                    help='remaining 2s data not used for pretesting')

parser.add_argument('--training-indices-3', 
                    type=str, 
                    default='training/index_files/indices_training-threes.npy', 
                    metavar='fname',
                    help='remaining 3s data not used for pretraining')
 
parser.add_argument('--testing-indices-3', 
                    type=str, 
                    default='evaluation/index_files/indices_testing-threes.npy', 
                    metavar='fname',
                    help='remaining 3s data not used for pretesting')

parser.add_argument('--training-indices-4', 
                    type=str, 
                    default='training/index_files/indices_training-fours.npy', 
                    metavar='fname',
                    help='remaining 4s data not used for pretraining') 

parser.add_argument('--testing-indices-4', 
                    type=str, 
                    default='evaluation/index_files/indices_testing-fours.npy', 
                    metavar='fname',
                    help='remaining 4s data not used for pretesting')

parser.add_argument('--training-indices-5', 
                    type=str, 
                    default='training/index_files/indices_training-fives.npy', 
                    metavar='fname',
                    help='remaining 5s data not used for pretraining') 

parser.add_argument('--testing-indices-5', 
                    type=str, 
                    default='evaluation/index_files/indices_testing-fives.npy', 
                    metavar='fname',
                    help='remaining 5s data not used for pretesting')

parser.add_argument('--training-indices-6', 
                    type=str, 
                    default='training/index_files/indices_training-sixes.npy', 
                    metavar='fname',
                    help='remaining 6s data not used for pretraining')
 
parser.add_argument('--testing-indices-6', 
                    type=str, 
                    default='evaluation/index_files/indices_testing-sixes.npy', 
                    metavar='fname',
                    help='remaining 6s data not used for pretesting')

parser.add_argument('--training-indices-7', 
                    type=str, 
                    default='training/index_files/indices_training-sevens.npy', 
                    metavar='fname',
                    help='remaining 7s data not used for pretraining') 

parser.add_argument('--testing-indices-7', 
                    type=str, 
                    default='evaluation/index_files/indices_testing-sevens.npy', 
                    metavar='fname',
                    help='remaining data not used for pretesting')

parser.add_argument('--training-indices-8', 
                    type=str, 
                    default='training/index_files/indices_training-eights.npy', 
                    metavar='fname',
                    help='remaining 8s data not used for pretraining') 

parser.add_argument('--testing-indices-8', 
                    type=str, 
                    default='evaluation/index_files/indices_testing-eigths.npy', 
                    metavar='fname',
                    help='remaining 8s data not used for pretesting')

parser.add_argument('--training-indices-9', 
                    type=str, 
                    default='training/index_files/indices_training-nines.npy', 
                    metavar='fname',
                    help='remaining data not used for pretraining') 

parser.add_argument('--testing-indices-9', 
                    type=str, 
                    default='evaluation/index_files/indices_testing-nines.npy', 
                    metavar='fname',
                    help='remaining data not used for pretesting')

parser.add_argument('--pretrained_0_model', 
                    type=str, 
                    default='evaluation/ms_vae_data_checkpoints/expert_checkpoints/semi-supervised/expert_vae_zeros_10.pt', 
                    help='')

parser.add_argument('--pretrained_1_model', 
                    type=str, 
                    default='evaluation/ms_vae_data_checkpoints/expert_checkpoints/semi-supervised/expert_vae_ones_10.pt', 
                    help='')

parser.add_argument('--pretrained_2_model', 
                    type=str, 
                    default='evaluation/ms_vae_data_checkpoints/expert_checkpoints/semi-supervised/expert_vae_twos_10.pt', 
                    help='')

parser.add_argument('--pretrained_3_model', 
                    type=str, 
                    default='evaluation/ms_vae_data_checkpoints/expert_checkpoints/semi-supervised/expert_vae_threes_10.pt', 
                    help='')

parser.add_argument('--pretrained_4_model', 
                    type=str, 
                    default='evaluation/ms_vae_data_checkpoints/expert_checkpoints/semi-supervised/expert_vae_fours_10.pt', 
                    help='')

parser.add_argument('--pretrained_5_model', 
                    type=str, 
                    default='evaluation/ms_vae_data_checkpoints/expert_checkpoints/semi-supervised/expert_vae_fives_10.pt', 
                    help='')

parser.add_argument('--pretrained_6_model', 
                    type=str, 
                    default='evaluation/ms_vae_data_checkpoints/expert_checkpoints/semi-supervised/expert_vae_sixes_10.pt', 
                    help='')

parser.add_argument('--pretrained_7_model', 
                    type=str, 
                    default='evaluation/ms_vae_data_checkpoints/expert_checkpoints/semi-supervised/expert_vae_sevens_10.pt', 
                    help='')

parser.add_argument('--pretrained_8_model', 
                    type=str, 
                    default='evaluation/ms_vae_data_checkpoints/expert_checkpoints/semi-supervised/expert_vae_eights_10.pt', 
                    help='')

parser.add_argument('--pretrained_9_model', 
                    type=str, 
                    default='evaluation/ms_vae_data_checkpoints/expert_checkpoints/semi-supervised/expert_vae_nines_10.pt', 
                    help='')

parser.add_argument('--m_samples', 
                    type=int, 
                    default=1,
                    help='Number of M samples used to approximate the integrals')

parser.add_argument('--freeze-decoder', 
                    default=True, 
                    type=lambda x: bool(strtobool(x)), 
                    help='True if the encoder weight should be frozen (not updated).')

parser.add_argument('--freeze-encoder', 
                    default=False, 
                    type=lambda x: bool(strtobool(x)), 
                    help='True if the decoder weight should be frozen (not updated).')

parser.add_argument('--unfreeze-decoders-after-100', 
                    default=False, 
                    type=lambda x: bool(strtobool(x)), 
                    help='True if the decoder weighs should be updated after 100 epoch of training.')

parser.add_argument('--unfreeze-decoders', 
                    default=False, 
                    type=lambda x: bool(strtobool(x)), 
                    help='True if the decoder weighs should be updated start of training.')

parser.add_argument('--unfreeze-decoder-1', 
                    default=False, 
                    type=lambda x: bool(strtobool(x)), 
                    help='True if the decoder 1 weights should updated.')

parser.add_argument('--unfreeze-decoder-2', 
                    default=False, 
                    type=lambda x: bool(strtobool(x)), 
                    help='True if the decoder 2 weights should updated.')

parser.add_argument('--pi-update-fixed', 
                    default=False, 
                    type=lambda x: bool(strtobool(x)), 
                    help='True if prior pi parameters should not be updated but kept fixed at generative value throughout the training.')

parser.add_argument('--log-interval', 
                    type=int, 
                    default=10, 
                    metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--dimz', 
                    type=int, 
                    default=20,
                    help='Dimension of latent space.')

parser.add_argument('--dimx', 
                    type=int, 
                    default=784,
                    help='Dimension of observable space.')

parser.add_argument('--pi-0-gen', 
                    type=float, 
                    default=0.1,
                    help='Generative pi for signal 0.')

parser.add_argument('--pi-1-gen', 
                    type=float, 
                    default=0.1,
                    help='Generative pi for signal 1.')

parser.add_argument('--pi-2-gen', 
                    type=float, 
                    default=0.1,
                    help='Generative pi for signal 2.')

parser.add_argument('--pi-3-gen', 
                    type=float, 
                    default=0.1,
                    help='Generative pi for signal 3.')

parser.add_argument('--pi-4-gen', 
                    type=float, 
                    default=0.1,
                    help='Generative pi for signal 4.')

parser.add_argument('--pi-5-gen', 
                    type=float, 
                    default=0.1,
                    help='Generative pi for signal 5.')

parser.add_argument('--pi-6-gen', 
                    type=float, 
                    default=0.1,
                    help='Generative pi for signal 6.')

parser.add_argument('--pi-7-gen', 
                    type=float, 
                    default=0.1,
                    help='Generative pi for signal 7.')

parser.add_argument('--pi-8-gen', 
                    type=float, 
                    default=0.1,
                    help='Generative pi for signal 8.')

parser.add_argument('--pi-9-gen', 
                    type=float, 
                    default=0.1,
                    help='Generative pi for signal 9.')

parser.add_argument('--pi-0-init', 
                    type=float, 
                    default=0.5,
                    help='Initial pi for signal 0.')

parser.add_argument('--pi-1-init', 
                    type=float, 
                    default=0.5,
                    help='Initial pi for signal 1.')

parser.add_argument('--pi-2-init', 
                    type=float, 
                    default=0.5,
                    help='Initial pi for signal 2.')

parser.add_argument('--pi-3-init', 
                    type=float, 
                    default=0.5,
                    help='Initial pi for signal 3.')

parser.add_argument('--pi-4-init', 
                    type=float, 
                    default=0.5,
                    help='Initial pi for signal 4.')

parser.add_argument('--pi-5-init', 
                    type=float, 
                    default=0.5,
                    help='Initial pi for signal 5.')

parser.add_argument('--pi-6-init', 
                    type=float, 
                    default=0.5,
                    help='Initial pi for signal 6.')

parser.add_argument('--pi-7-init', 
                    type=float, 
                    default=0.5,
                    help='Initial pi for signal 7.')

parser.add_argument('--pi-8-init', 
                    type=float, 
                    default=0.5,
                    help='Initial pi for signal 8.')

parser.add_argument('--pi-9-init', 
                    type=float, 
                    default=0.5,
                    help='Initial pi for signal 9.')

parser.add_argument('--H-dim', 
                    type=int, 
                    default=10,
                    help='Hidden space dimension of discrete latents s.')

parser.add_argument('--n-train', 
                    type=int, 
                    default=50000,
                    help='Number of training samples to generate.')

parser.add_argument('--n-test', 
                    type=int, 
                    default=5000,
                    help='Number of testing samples to generate.')

parser.add_argument('--num-workers', 
                    type=int, 
                    default=4,
                    help='Number of data loading workers (parallel).')

parser.add_argument('--prior', 
                    type=str, 
                    default='gauss',
                    help='Prior over latent variables')

parser.add_argument('--scale', 
                    type=float, 
                    default=0.1,
                    help='Scale of Laplace likelihood')

parser.add_argument('--scale-init', 
                    type=float, 
                    default=1,
                    help='Inital Scale of Laplace likelihood')

parser.add_argument('--learning-rate', 
                    type=float,
                    default=.0002,
                    help='Learning rate') 

parser.add_argument('--variational', 
                    type=str, 
                    default=True,
                    help='VAE if True, AE if False.')

parser.add_argument('--decay', 
                    type=float, 
                    default=.9998,
                    help='Decay rate of scheduler for optimizer.')

parser.add_argument('--beta-max', 
                    type=float, 
                    default=0.5,
                    help='beta-VAE max value of weight')

parser.add_argument('--epochs', 
                    type=int, 
                    default=101,
                    help='Training epochs') 

parser.add_argument('--warm-up', 
                    type=int, 
                    default=50,
                    help='Number of iterations to warm up beta.') # not used

parser.add_argument('--save-interval', 
                    type=int, 
                    default=1,
                    help='How many epochs to wait before saving model.')

parser.add_argument('--save-reco-every', 
                    type=int, 
                    default=1,
                    help='Save reconstruction every ... epoch.')  
