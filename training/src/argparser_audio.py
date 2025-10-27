import argparse
from distutils.util import strtobool


parser = argparse.ArgumentParser(description='MS-VAE')

parser.add_argument('--batch-size', 
                    type=int, 
                    default=64, metavar='N',
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

parser.add_argument('--unfreeze-decoders-after-n', 
                    default=True, 
                    type=lambda x: bool(strtobool(x)), 
                    help='True if the decoder weighs should be updated after n epoch of training.')

parser.add_argument('--unfreeze-n', 
                    type=int, 
                    default=100,
                    help='unfreeze decoders after n epochs')

parser.add_argument('--label-file-train', 
                    type=str, 
                    default='/frame_labels_train.csv',
                    help='labels for the train section')

parser.add_argument('--label-file-test', 
                    type=str, 
                    default='/frame_labels_val.csv',
                    help='labels for the test section')

parser.add_argument('--train-dir-mixture', 
                    type=str, 
                    default='/train-mixtures/', 
                    metavar='fname',
                    help='Folder containing the training mixture data')

parser.add_argument('--test-dir-mixture', 
                    type=str, 
                    default='/test-mixtures/', 
                    metavar='fname',
                    help='Folder containing the testing mixture data')

parser.add_argument('--winlen', 
                    type=int, 
                    default=512,
                    help='STFT window length')

parser.add_argument('--hopfrac', 
                    type=int, 
                    default=256,
                    help='STFT hop length')

parser.add_argument('--fs', 
                    type=int, 
                    default=16000,
                    help='sampling frequency')

parser.add_argument('--nfft', 
                    type=int, 
                    default=512,
                    help='FFT length, number of bins')

parser.add_argument('--m_samples', 
                    type=int, 
                    default=1,
                    help='Number of M samples used to approximate the integrals')

parser.add_argument('--pi-update-fixed', 
                    default=False, 
                    type=lambda x: bool(strtobool(x)), 
                    help='True if prior pi parameters should not be updated but kept fixed at generative value throughout the training.')

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

parser.add_argument('--scale', 
                    type=float, 
                    default=0.0008,
                    help='Scale of Laplace likelihood')

parser.add_argument('--scale-init', 
                    type=float, 
                    default=1,
                    help='Inital Scale of Laplace likelihood')

parser.add_argument('--pretrained_speaker_0_model', 
                    type=str, 
                    default='/speaker-0/model_vae_K1.pt', 
                    help='')

parser.add_argument('--pretrained_speaker_1_model', 
                    type=str, 
                    default='/speaker_1/model_vae_K1.pt', 
                    help='')

parser.add_argument('--pretrained_speaker_2_model', 
                    type=str, 
                    default='/speaker_2/model_vae_K1.pt', 
                    help='')

parser.add_argument('--save-path', 
                    type=str, 
                    default='/training-saves/', 
                    metavar='fname',
                    help='Folder for saving') 

parser.add_argument('--log-interval', 
                    type=int, 
                    default=10, 
                    metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--sources', 
                    type=int, 
                    default=3,
                    help='Number of sources to infer')

parser.add_argument('--dimz', 
                    type=int, 
                    default=20,
                    help='Dimension of latent space.') 

parser.add_argument('--num-workers', 
                    type=int, 
                    default=4,
                    help='Number of data loading workers (parallel).')

parser.add_argument('--prior', 
                    type=str, 
                    default='gauss',
                    help='Prior over latent variables')

parser.add_argument('--learning-rate', 
                    type=float, 
                    default=.0002,
                    help='Learning rate')

parser.add_argument('--variational', 
                    type=str, 
                    default=True,
                    help='VAE if True, AE if False.')

parser.add_argument('-un', '--freeze-decoder', 
                    default=True, 
                    type=lambda x: bool(strtobool(x)), 
                    help='True if the decoder weight should be frozen (not updated).')

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
                    default=151,
                    help='Training epochs') 

parser.add_argument('--warm-up', 
                    type=int, 
                    default=50,
                    help='Number of iterations to warm up beta.') 

parser.add_argument('--save-interval', 
                    type=int, 
                    default=1,
                    help='How many epochs to wait before saving model.')

parser.add_argument('--save-reco-every', 
                    type=int, 
                    default=5,
                    help='Save reconstruction every ... epoch.')
