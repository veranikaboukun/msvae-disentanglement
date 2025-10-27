import argparse
from distutils.util import strtobool


''' 
Argument parser for model pretraining and evaluation 
'''

parser = argparse.ArgumentParser(description='Pretraining MS-VAE expert decoders on MNIST data')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Random seed (default: 1)')

parser.add_argument('--data-directory', type=str, default='/MNIST/', metavar='fname',
                    help='Folder containing the data')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='How many batches to wait before logging training status')

parser.add_argument('--log-labels', default=False, type=lambda x: bool(strtobool(x)),
                    help='Save "labels" (learned mu_z and logvar_z in the end of the training) for every image')

parser.add_argument('--sources', type=int, default=1,
                    help='Number of sources to infer')

parser.add_argument('--dimz', type=int, default=20,
                    help='Dimension of latent space.')

parser.add_argument('--subset', type=int, default=0,
                    help='MNIST index subset')

parser.add_argument('--percent_labeled', type=float, default=0.1,
                    help='Percent of total training data used for pretraining of the model, in decimal format')

parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of data loading workers (parallel)')

parser.add_argument('--prior', type=str, default='gauss',
                    help='Prior over latent variables')

parser.add_argument('--scale', type=float, default=0.5,
                    help='Scale parameter of Laplace likelihood')

parser.add_argument('--variational', type=str, default=True,
                    help='VAE if True, AE if False.')

parser.add_argument('--m_samples', type=int, default=1,
                    help='Number of M samples used to approximate the integrals in ELBO')

parser.add_argument('--learning-rate', type=float, default=.0001,
                    help='Learning rate')

parser.add_argument('--decay', type=float, default=.9998,
                    help='Decay rate of scheduler for optimizer.')

parser.add_argument('--beta-max', type=float, default=0.5,
                    help='beta-VAE max value of weight')

parser.add_argument('--epochs', type=int, default=10,
                    help='Training epochs')

parser.add_argument('--warm-up', type=int, default=100,
                    help='Number of iterations to warm up beta.')

parser.add_argument('-un', '--freeze-decoder', default=False, type=lambda x: bool(strtobool(x)), 
                    help='True if the decoder weight should be frozen (not updated).')

parser.add_argument('--save-path', type=str, default='/pretrained_mnist_models/', metavar='fname',
                    help='Folder for saving the reconstructions') 

parser.add_argument('--save-interval', type=int, default=1,
                    help='How many epochs to wait before saving model.')

parser.add_argument('--save-reco-every', type=int, default=1,
                    help='Save reconstructed image every ... epoch.')  