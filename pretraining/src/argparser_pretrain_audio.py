import argparse


''' 
Argument parser for model pretraining and evaluation 
'''

parser = argparse.ArgumentParser(description='Pretraining MS-VAE expert decoders on acoustic data')

parser.add_argument('--batch-size', type=int, default=9, metavar='N',
                    help='Input batch size for training')

parser.add_argument('--batch-size-test', type=int, default=23, metavar='N',
                    help='Input batch size for testing')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')

parser.add_argument('--percent_labeled', type=float, default=0.1,
                    help='Percent of total training data used for pretraining of the model, in decimal point')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--train-dir', type=str, default='', metavar='fname',
                    help='Folder containing the train data')

parser.add_argument('--valid-dir', type=str, default='', metavar='fname',
                    help='Folder containing the test data')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='How many batches to wait before logging training status')

parser.add_argument('--sources', type=int, default=1,
                    help='Number of sources to infer')

parser.add_argument('--dimz', type=int, default=20,
                    help='Dimension of latent space')

parser.add_argument('--winlen', type=int, default=512,
                    help='STFT window length')

parser.add_argument('--hopfrac', type=int, default=256,
                    help='STFT hop length')

parser.add_argument('--fs', type=int, default=44100,
                    help='Sampling frequency')

parser.add_argument('--nfft', type=int, default=512,
                    help='FFT length, number of bins')

parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of data loading workers (parallel).')

parser.add_argument('--prior', type=str, default='gauss',
                    help='Prior over latent variables')

parser.add_argument('--scale', type=float, default=0.5,
                    help='Scale of Laplace likelihood')

parser.add_argument('--variational', type=str, default=True,
                    help='VAE if True, AE if False')

parser.add_argument('--m_samples', type=int, default=1,
                    help='Number of M samples used to approximate the integrals')

parser.add_argument('--learning-rate', type=float, default=.0001,
                    help='Learning rate')

parser.add_argument('--decay', type=float, default=.9998,
                    help='Decay rate of scheduler for optimizer')

parser.add_argument('--beta-max', type=float, default=0.5,
                    help='beta-VAE max value of weight')

parser.add_argument('--epochs', type=int, default=10,
                    help='Training epochs')

parser.add_argument('--warm-up', type=int, default=100,
                    help='Number of iterations to warm up beta.')

parser.add_argument('--save-path', type=str, default='pretraining_audio/', metavar='fname',
                    help='Folder for saving the reconstructions') 

parser.add_argument('--save-interval', type=int, default=1,
                    help='How many epochs to wait before saving model')

parser.add_argument('--save-reco-every', type=int, default=1,
                    help='Save reconstruction every ... epoch.')  