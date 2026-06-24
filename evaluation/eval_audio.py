# Copyright (C) 2026 Veranika Boukun (veranika.boukun@uni-oldenburg.de), Jörg Lücke

import numpy as np
import torch
import itertools 
from torch import nn
import pandas as pd
from src.dataload_audio import SpectrogramDataset, SpectrogramDatasetMetrics
from src.model_streams import MultiStream_VAE_audio, Loss_k_sources
from src.utils import accuracy_score
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import glob
from torch.distributions import Laplace
import time
import argparse
from distutils.util import strtobool
import mir_eval
from torchaudio.transforms import Spectrogram
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(1)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}


def is_active_by_center(t_start, t_end, intervals):
    center = 0.5 * (t_start + t_end)
    return any(start <= center < end for start, end in intervals)

def compute_si_sdr(reference, estimated):
    reference_mean = np.mean(reference)
    estimated_mean = np.mean(estimated)
    
    # Normalizing connection
    reference_norm = reference - reference_mean
    estimated_norm = estimated - estimated_mean
    
    # Compute true scaling factor
    alpha = np.dot(estimated_norm, reference_norm) / np.dot(reference_norm, reference_norm)
    
    # Scaled reference approximation
    reference_approx = alpha * reference_norm
    
    # Compute SI-SDR
    si_sdr_value = 10 * np.log10(np.sum(reference_approx ** 2) / np.sum((estimated_norm - reference_approx) ** 2))
    
    return si_sdr_value

def run_inference(model):
    
    _, n_frames_full = D1.shape
    full_mag_sp1 = np.zeros((f, n_frames_full))
    full_mag_sp2 = np.zeros((f, n_frames_full))
    full_mag_mix = np.zeros((f, n_frames_full))
    frame_counter = np.zeros((f, n_frames_full))

    q_s1_x_full = np.zeros((n_frames_full))
    q_s2_x_full = np.zeros((n_frames_full))
    frame_counter_q_s = np.zeros((n_frames_full))
    
    model.eval()
    accuracies = torch.zeros(1)

    mode_vals_pred = []
    mode_vals_label = []

    time_intervals = []

    current_time = 0.0
    # Start timer
    start_time_counter = time.time()

    with torch.no_grad():
        for i, (data, labels) in enumerate(spectrogram_dataloader_test):
            
            data = data.to(device).view(-1, dimx)
            data = data.float()

            mu_theta_list, mu_phi_list, logvar_phi_list, z_list = model(data, M=m_samples)
            _, _, _, _, _, _, q_s_x = loss_overall(data, mu_theta_list, 
                                                    mu_phi_list, logvar_phi_list, 
                                                    s_list, pi_values, z_list, 
                                                    beta=1, N=batch_size, 
                                                    M=m_samples, scale=b)

            # individual s1/s2 posterior probabilities
            # s1 is q(case2) + q(case3)
            q_s1_x = q_s_x[..., 2] + q_s_x[..., 3]

            # s2 is q(case1) + q(case3)
            q_s2_x = q_s_x[..., 1] + q_s_x[..., 3]

            # now computing the accuracy correctly
            labels_speakers =labels[:, 1:]

            _, target = torch.max(s_list.detach().cpu() @ labels_speakers.T.long(), dim=0) 

            mask1 = q_s1_x > args.theta1  
            mask2 = q_s2_x > args.theta2  
            prediction = (mask1.int() * 2 + mask2.int() * 1).detach().cpu()

            accuracy = accuracy_score(target.squeeze(0), prediction)

            accuracies[0] += round(accuracy, 2)

            mu_theta_i_list = []
            for i in range(sources):
                mu_theta_i = mu_theta_list[i].sum(1).detach().cpu() # check dim!
                mu_theta_i_list.append(mu_theta_i)

            # Predictions on interval
            unique_vals_pred, counts_pred = torch.unique(prediction, return_counts=True) 
            max_count_idx_pred = torch.argmax(counts_pred)
            mode_val_pred = unique_vals_pred[max_count_idx_pred].item()
            mode_vals_pred.append(mode_val_pred)

            # Target activations on an interval
            unique_vals_label, counts_label = torch.unique(target, return_counts=True)
            max_count_idx_label = torch.argmax(counts_label)
            mode_val_label = unique_vals_label[max_count_idx_label].item()
            mode_vals_label.append(mode_val_label)

            # Save the interval
            time_intervals.append((current_time, current_time + seconds_per_batch))
            current_time += seconds_per_batch

            mu_theta_i_new_list = []
            
            for i, mu_theta_i in enumerate(mu_theta_i_list):
                
                mu_theta_i_new = torch.zeros_like(mu_theta_i).to(device)
                for n, index in enumerate(prediction):

                    s_vector = s_list[index].detach().cpu().float() 
                    s_i = s_vector[i]
                    mu_theta_i_new[n, ...] = mu_theta_i[n, ...] * s_i  

                mu_theta_i_new_list.append(mu_theta_i_new)

            recon_y = torch.stack(mu_theta_i_new_list, dim=2)
            recons = recon_y.sum(2) 

            relative_indices = labels[:, 0].cpu().numpy()
            frame_indices = [rel_idx * stride for rel_idx in relative_indices]
            
            # Start OLA (overlap-add) applied to spectrogram frames 
            for n, start in enumerate(frame_indices):
                start = int(start)
                end = start + t
                sp1_patch = recon_y[n, :, 0].cpu().numpy().reshape(f, t)
                sp2_patch = recon_y[n, :, 1].cpu().numpy().reshape(f, t)
                mix_patch = recons[n, :].cpu().numpy().reshape(f, t)

                q_s1_x_patch = q_s1_x[n].cpu().numpy()
                q_s2_x_patch = q_s2_x[n].cpu().numpy()

                if end > n_frames_full:
                    overlap = n_frames_full - start

                    if overlap > 0:
                        full_mag_sp1[:, start:n_frames_full] += sp1_patch[:, :overlap]
                        full_mag_sp2[:, start:n_frames_full] += sp2_patch[:, :overlap]
                        full_mag_mix[:, start:n_frames_full] += mix_patch[:, :overlap]

                        q_s1_x_full[start:n_frames_full] += q_s1_x_patch
                        q_s2_x_full[start:n_frames_full] += q_s2_x_patch

                        frame_counter[:, start:n_frames_full] += 1
                        frame_counter_q_s[start:n_frames_full] += 1
                    
                else:
                    full_mag_sp1[:, start:end] += sp1_patch
                    full_mag_sp2[:, start:end] += sp2_patch
                    full_mag_mix[:, start:end] += mix_patch

                    q_s1_x_full[start:end] += q_s1_x_patch
                    q_s2_x_full[start:end] += q_s2_x_patch

                    frame_counter[:, start:end] += 1
                    frame_counter_q_s[start:end] += 1

        frame_counter[frame_counter == 0] = 1
        full_mag_sp1 /= frame_counter
        full_mag_sp2 /= frame_counter
        full_mag_mix /= frame_counter
        q_s1_x_full /= frame_counter_q_s
        q_s2_x_full /= frame_counter_q_s

    # End timer
    end_time_counter = time.time()

    # Calculate elapsed time
    inference_time = end_time_counter - start_time_counter
    print(f"Inference Time (Waveform reconstruction): {inference_time:.4f} seconds")

    accuracies /= len(spectrogram_dataloader_test)
    print('====> Test set accuracy: {:.4f}'.format(accuracies[0]), flush=False)

    return mode_vals_label, mode_vals_pred, full_mag_mix, full_mag_sp1, full_mag_sp2, time_intervals, q_s1_x_full, q_s2_x_full

def run_inference_metrics(model):

    model.eval()

    average_metrics = torch.zeros(6)
    len_active = 0
    total_num_active = 0

    # Start timer
    start_time_counter = time.time()

    with torch.no_grad():
        for i, (data, phase_mix, wave_1, wave_2, labels) in enumerate(spectrogram_dataloader_test):
            
            data = data.to(device).view(-1, dimx)
            data = data.float()

            active_sources = torch.all(labels[:, 1:] == 1, dim=1)
            
            if torch.any(active_sources):
                
                active_data = data[active_sources]
                n_active, _ = active_data.shape

                active_wave_1 = wave_1[active_sources]
                active_wave_2 = wave_2[active_sources]

                if n_active > 1:
                    mu_theta_list, mu_phi_list, logvar_phi_list, z_list = model(active_data, M=m_samples)
                    _, _, _, _, _, _, q_s_x = loss_overall(active_data, mu_theta_list, 
                                                            mu_phi_list, logvar_phi_list, 
                                                            s_list, pi_values, z_list, 
                                                            beta=1, N=batch_size, 
                                                            M=m_samples, scale=b)

                
                    _, prediction = torch.max(q_s_x.detach().cpu(), dim=1)

                    mu_theta_i_list = []
                    for i in range(sources):
                        mu_theta_i = mu_theta_list[i].sum(1).detach().cpu()
                        mu_theta_i_list.append(mu_theta_i)


                    mu_theta_i_new_list = []
                    estimated_sources_batch = []
                    
                    for i, mu_theta_i in enumerate(mu_theta_i_list):
                    
                        mu_theta_i_new = torch.zeros_like(mu_theta_i).to(device)
                        spec_sample_i = torch.zeros_like(mu_theta_i).to(device)
                        estimated_sources_i = np.zeros_like(active_wave_1)

                        for n, index in enumerate(prediction): 
                            s_vector = s_list[index].detach().cpu().float() 
                            s_i = s_vector[i]
                        
                            mu_theta_i_new[n, ...] = mu_theta_i[n, ...] * s_i

                            spec_sample_i[n, ...] = Laplace(mu_theta_i_new[n, ...], args.b).sample()

                            reconstructed_i = spec_sample_i[n, ...].reshape(f, t).detach().cpu().numpy() * np.exp(1j * phase_mix[n, ...].detach().cpu().numpy())
                            # convert to wave
                            estimated_sources_i[n, ...] = librosa.istft(reconstructed_i, hop_length=hop_length)

                        mu_theta_i_new_list.append(mu_theta_i_new)
                        estimated_sources_batch.append(estimated_sources_i)

                    reference_sources_batch = [active_wave_1, active_wave_2]

                    ref_np = np.stack([x.cpu().numpy() for x in reference_sources_batch], axis=1)   
                    est_np = np.stack([x for x in estimated_sources_batch], axis=1)    

                    sir_list_1 = []
                    sir_list_2 = []
                    sar_list_1 = []
                    sar_list_2 = []
                    si_sdr_list_1 = []
                    si_sdr_list_2 = []
                    for j in range(n_active):
                        refs_j = ref_np[j]  
                        ests_j = est_np[j]  
                        # compute metrics
                        _, sir, sar, _ = mir_eval.separation.bss_eval_sources(refs_j, ests_j)
                        si_sdr_1 = compute_si_sdr(refs_j[0], ests_j[0])
                        si_sdr_2 = compute_si_sdr(refs_j[1], ests_j[1])
                        sir_list_1.append(sir[0]) 
                        sir_list_2.append(sir[1]) 
                        sar_list_1.append(sar[0])
                        sar_list_2.append(sar[1])
                        si_sdr_list_1.append(si_sdr_1)
                        si_sdr_list_2.append(si_sdr_2)

                    average_metrics[0] += round(np.mean(si_sdr_list_1), 2)
                    average_metrics[1] += round(np.mean(si_sdr_list_2), 2)
                    average_metrics[2] += round(np.mean(sir_list_1), 2)
                    average_metrics[3] += round(np.mean(sir_list_2), 2)
                    average_metrics[4] += round(np.mean(sar_list_1), 2)
                    average_metrics[5] += round(np.mean(sar_list_2), 2)

                        
                    len_active += 1
                    total_num_active += n_active

            else:
                print('Evaluation of source separation metrics only possible if both sources are active (non-zero). Returned metrics are all zeros.', flush=True)

    # End timer
    end_time_counter = time.time()

    # Calculate elapsed time
    inference_time = end_time_counter - start_time_counter
    print(f"Inference Time (Metrics): {inference_time:.4f} seconds")

    average_metrics /= len_active
    print('====> Average SI-SDR source 1: {:.4f}'.format(average_metrics[0]), flush=False)
    print('====> Average SI-SDR source 2: {:.4f}'.format(average_metrics[1]), flush=False)
    print('====> Average SIR source 1: {:.4f}'.format(average_metrics[2]), flush=False)
    print('====> Average SIR source 2: {:.4f}'.format(average_metrics[3]), flush=False)
    print('====> Average SAR source 1: {:.4f}'.format(average_metrics[4]), flush=False)
    print('====> Average SAR source 2: {:.4f}'.format(average_metrics[5]), flush=False)
    
    return average_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test-dir', type=str, 
                        default='/test/mixture/')
    
    parser.add_argument('--test-dir-s1', type=str, 
                        default='/test/speaker-1/')
    
    parser.add_argument('--test-dir-s2', type=str, 
                        default='/test/speaker-2/')
    
    parser.add_argument('--model-path', type=str, 
                        default='/training/audio/MS-VAE-model-1.pt') # checkpoint for a trained MS-VAE model 
    
    parser.add_argument('--model-path-2', type=str, 
                    default='/training/audio/MS-VAE-model-2.pt') # checkpoint of an additional MS-VAE model (optional)

    parser.add_argument('--winlen', type=int, default=512,
                    help='STFT window length')
    
    parser.add_argument('--fs', type=int, default=48000,
                    help='sampling frequency')
    
    parser.add_argument('--sources', type=int, default=2,
                    help='Number of sources to infer')
    
    parser.add_argument('--dimz', type=int, default=20,
                    help='Dimension of latent space.')
    
    parser.add_argument('--batch-size', type=int, default=23, metavar='N',
                    help='input batch size of 23 corresponds to 1 seconds')
    
    parser.add_argument('--seconds-per-batch', type=float, default=1.0,
                    help='how many seconds does the batch correspond to')
    
    parser.add_argument('--label-file-test', type=str, 
                    default='/test/mixture/frame_labels_test.csv',
                    help='labels for the test section')
    
    parser.add_argument('--orig-sp1-path', type=str, 
                    default='/test/audio/speaker_1.wav',
                    help='path to source 1 audio')
    
    parser.add_argument('--orig-sp2-path', type=str, 
                    default='test/audio/speaker_2.wav',
                    help='path to source 2 audio')
    
    parser.add_argument('--orig-test-mix', type=str, 
                    default='/test/audio/mix.wav',
                    help='path to source mixture')

    parser.add_argument('--pi-1', type=float, default=0.30262,
                    help='learned pi_1 for source 1.')
    parser.add_argument('--pi-2', type=float, default=0.36996,
                        help='learned pi_2 for source 1.')
    parser.add_argument('--b', type=float, default=0.0010854,
                        help='learned scale for source 1')
    
    parser.add_argument('--pi-1-2', type=float, default=0.30262,
                    help='learned pi_1 for source 2.')
    parser.add_argument('--pi-2-2', type=float, default=0.36996,
                        help='learned pi_2 for source 2.')
    parser.add_argument('--b-2', type=float, default=0.0010854,
                        help='learned scale for source 2')
    
    parser.add_argument('--theta1', type=float, default=0.0,
                    help='Thershold posterior probability of source 1.')
    
    parser.add_argument('--theta2', type=float, default=0.0,
                        help='Thershold posterior probability of source 2.')
    
    parser.add_argument('--save-path', type=str, 
                        default='s/evaluation/')
    
    parser.add_argument('--plot-spec-separate', default=False, type=lambda x: bool(strtobool(x)), 
                    help='True if separate speech spectrograms should be plotted.')
    
    parser.add_argument('--plot-spec-mixture', default=False, type=lambda x: bool(strtobool(x)), 
                    help='True if mixture speech spectrograms with decision boundaries should be plotted.')
    
    parser.add_argument('--save-waves', default=False, type=lambda x: bool(strtobool(x)), 
                    help='True if individual and mixtures waveforms should be saved.')
    
    parser.add_argument('--plot-supervised-semi-supervised', default=False, type=lambda x: bool(strtobool(x)), 
                    help='True if supervised and semi-supervised models are compared.')
    
    parser.add_argument('--compute-metrics-per-frame', default=False, type=lambda x: bool(strtobool(x)), 
                    help='True if compute for every frame SI-SDR, SIR, SAR.')
    
    parser.add_argument('--plot-probabilities-per-frame', default=False, type=lambda x: bool(strtobool(x)), 
                    help='True if plot individual source posterior probabilities per frame. ')
    
    parser.add_argument('--compute-der', default=True, type=lambda x: bool(strtobool(x)), 
                    help='True if compute the diarization error rate. ')
       
    

    args = parser.parse_args()

    winlen= args.winlen #512
    n_fft = 512
    hop_length = 256

    fs= args.fs #48000

    sources = args.sources #2
    dimz = args.dimz #20
    
    # for 1 second:
    batch_size = args.batch_size #23 
    seconds_per_batch = args.seconds_per_batch #1.0

    # for exampla, 2.5 seconds correspond to:
    # batch_size = 57
    # seconds_per_batch = 2.5

    m_samples = 1
    freeze_decoder = True
    variational = True

    H = sources
    s_list = torch.asarray(list(itertools.product(range(2), repeat=H))).to(device)

    test_path = args.test_dir
    valid_files_mix = sorted(glob.glob(test_path + '*.wav', recursive=True))
    label_file_test = args.label_file_test
    labels_test = pd.read_csv(label_file_test).values

    spec_transform = Spectrogram(n_fft=n_fft, 
                                win_length=winlen, 
                                hop_length=hop_length, 
                                pad_mode='constant')

    test_dataset_mix = SpectrogramDataset(valid_files_mix, 
                                            winlen, 
                                            transform=spec_transform, 
                                            max_timesteps=17, 
                                            transpose=False)

    # Replace with zeros those frames which are labeled with (0, 0)
    test_data = []
    scale = 0.0008 # small scale to be non-zero
    for idx in range(len(test_dataset_mix)):
        spec = test_dataset_mix[idx]
        label = labels_test[idx]

        if label[1] == 0 and label[2] == 0:    
            spec = torch.zeros_like(spec)
            # run-6
            spec = Laplace(spec, scale).sample()
        test_data.append(spec)

    test_data_tensor = torch.stack(test_data)
    labels_test = torch.Tensor(labels_test)

    test_set = TensorDataset(test_data_tensor, labels_test)

    spectrogram_dataloader_test = torch.utils.data.DataLoader(test_set, 
                                                            batch_size=batch_size, 
                                                            shuffle=False) # TODO: NEVER SHUFFLE TRUE!!!!!

    f, t = test_set[0][0].shape
    dimx = int(t*f)
    D = dimx
    stride=t//2
    length = len(test_set)

    cutoff = 120*fs # 2min cutoff

    orig_sp1, sr = sf.read(args.orig_sp1_path)
    orig_sp2, _ = sf.read(args.orig_sp2_path)

    orig_sp1 = orig_sp1[:cutoff]
    orig_sp2 = orig_sp2[:cutoff]
    orig_mix = orig_sp1 + orig_sp2

    D1 = librosa.stft(orig_sp1, n_fft=n_fft, hop_length=hop_length)
    D2 = librosa.stft(orig_sp2, n_fft=n_fft, hop_length=hop_length)
    Dmix = librosa.stft(orig_mix, n_fft=n_fft, hop_length=hop_length)

    S1 = np.abs(D1)
    S2 = np.abs(D2)
    Smix = np.abs(Dmix)

    S1_dB = librosa.amplitude_to_db(S1, ref=np.max)
    S2_dB = librosa.amplitude_to_db(S2, ref=np.max)
    Smix_dB = librosa.amplitude_to_db(Smix, ref=np.max)


    model = MultiStream_VAE_audio(dimx=dimx, dimz=dimz, 
                                        n_sources=sources, device=device, 
                                        variational=variational, 
                                        freeze_decoder=freeze_decoder)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    model.to(device)

    # model.load_state_dict(torch.load(args.model_path)) # gold12/13
    model.module.load_state_dict(torch.load(args.model_path)) # gold15
    pi_values = [args.pi_1, args.pi_2]
    b = args.b
    save_path = args.save_path


    loss_overall = Loss_k_sources(sources=sources, 
                                    likelihood='laplace', 
                                    variational=variational, 
                                    prior='gauss') 

    mode_vals_label, mode_vals_pred, full_mag_mix, full_mag_sp1, full_mag_sp2, time_intervals, q_s1_x_full, q_s2_x_full = run_inference(model)


    S3_dB = librosa.amplitude_to_db(full_mag_sp1, ref=np.max)
    S4_dB = librosa.amplitude_to_db(full_mag_sp2, ref=np.max)
    S5_dB = librosa.amplitude_to_db(full_mag_mix, ref=np.max)

    min_frames = min(S1_dB.shape[1], S2_dB.shape[1], S3_dB.shape[1], S4_dB.shape[1])
    S1_dB = S1_dB[:, :min_frames]
    S2_dB = S2_dB[:, :min_frames]
    S3_dB = S3_dB[:, :min_frames]
    S4_dB = S4_dB[:, :min_frames]


    # calculate theta1 and theta2 for improved accuracy/prediction
    theta1 = np.quantile(q_s1_x_full, 1 - args.pi_1)
    theta2 = np.quantile(q_s2_x_full, 1 - args.pi_2)
    print(f"Theta for arr1 (pi1=0.3): {theta1:.4f}")
    print(f"Theta for arr2 (pi2=0.37): {theta2:.4f}") # both are cutoff at ~0!

    # Check: This should be close to pi1/pi2
    A1 = (q_s1_x_full > theta1).sum()
    C1 = len(q_s1_x_full)

    A2 = (q_s2_x_full > theta2).sum()
    C2 = len(q_s2_x_full)
    print(f"A1/C1: {A1/C1:.4f}  (should be ~{args.pi_1})")
    print(f"A2/C2: {A2/C2:.4f}  (should be ~{args.pi_2})")

    if args.plot_probabilities_per_frame: 
        t = np.arange(len(q_s1_x_full)) * hop_length / fs  # makes time go from 0 to end_time

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Plot each line with a color and label
        axes[0].plot(t, q_s1_x_full, color='tab:blue')
        axes[0].set_ylabel("Posterior probability value")
        axes[0].set_xlim(0, min_frames * hop_length / fs)
        axes[0].set_ylim(0, 1.2)
        axes[0].set_title("Posterior probability q(s1|x)")

        # Plot second source
        axes[1].plot(t, q_s2_x_full, color='tab:orange')
        axes[1].set_ylabel("Posterior probability value")
        axes[1].set_xlim(0, min_frames * hop_length / fs)
        axes[1].set_ylim(0, 1.2)
        axes[1].set_title("Posterior probability q(s2|x)")

        axes[2].plot(t, q_s1_x_full, color='tab:blue', label='q(s1|x)')
        axes[2].plot(t, q_s2_x_full, color='tab:orange', label='q(s2|x)')
        axes[2].set_ylabel("Posterior probability value")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_xlim(0, min_frames * hop_length / fs)
        axes[2].set_ylim(0, 1.2)
        axes[2].set_title("Posterior probability q(s1|x) and q(s2|x) (Overlay)")
        axes[2].legend()

        plt.tight_layout()
        fig.savefig(save_path + 'probabilities_per_frame.png')
        plt.close()

    if args.plot_spec_separate:
        # Plot spectrograms - color
        fig, axes = plt.subplots(2, 2, figsize=(12, 4.8), sharex=True)

        librosa.display.specshow(S1_dB, sr=fs, n_fft=512, hop_length=256, x_axis='time', y_axis='log', ax=axes[0, 0])
        axes[0, 0].set_title("Orig. Speaker 1")

        librosa.display.specshow(S3_dB, sr=fs, n_fft=512, hop_length=256, x_axis='time', y_axis='log', ax=axes[1, 0])
        axes[1, 0].set_title("Est. Speaker 1")

        librosa.display.specshow(S2_dB, sr=fs, n_fft=512, hop_length=256, x_axis='time', y_axis='log', ax=axes[0, 1])
        axes[0, 1].set_title("Orig. Speaker 2")

        librosa.display.specshow(S4_dB, sr=fs, n_fft=512, hop_length=256, x_axis='time', y_axis='log', ax=axes[1, 1])
        axes[1, 1].set_title("Est. Speaker 2")

        end_time = min_frames * hop_length / fs
        for ax_row in axes:  
            for ax in ax_row:  
                ax.set_xlim(0, end_time)

        plt.tight_layout()
        fig.savefig(save_path + 'speakers_prediction_vs_orig.png')
        plt.close()

    if args.plot_spec_mixture: 
        # Find where the case changes for prediction
        change_times_pred = []
        cases_before_pred = []
        cases_after_pred = []

        for i in range(1, len(mode_vals_pred)):
            if mode_vals_pred[i] != mode_vals_pred[i-1]:
                change_times_pred.append(time_intervals[i][0])  # start time of interval where change occurs
                cases_before_pred.append(mode_vals_pred[i-1])
                cases_after_pred.append(mode_vals_pred[i])

        # Find where the case changes for target
        change_times_label = []
        cases_before_label = []
        cases_after_label = []

        for i in range(1, len(mode_vals_label)):
            if mode_vals_label[i] != mode_vals_label[i-1]:
                change_times_label.append(time_intervals[i][0])  
                cases_before_label.append(mode_vals_label[i-1])
                cases_after_label.append(mode_vals_label[i])

        # Regular plot - color
        fig, axes = plt.subplots(2, 1, sharex=True)

        librosa.display.specshow(Smix_dB, sr=fs, n_fft=512, hop_length=256, x_axis='time', y_axis='log', ax=axes[0])
        axes[0].set_title("Orig. Speaker Mixture")

        librosa.display.specshow(S5_dB, sr=fs, n_fft=512, hop_length=256, x_axis='time', y_axis='log', ax=axes[1])
        axes[1].set_title("Est. Speaker Mixture")

        end_time = min_frames * hop_length / fs

        for ax in axes:
            ax.set_xlim(0, end_time)

        for t, before, after in zip(change_times_label, cases_before_label, cases_after_label):
            for ax in [axes[0]]:
                ax.axvline(x=t, color='red', linestyle='--', linewidth=1.5)
                ax.text(t, ax.get_ylim()[1]*0.98, f"{before}→{after}", color='red', rotation=90, va='top', ha='right')

        # Plot prediction change points in green on axes 1 and 3
        for t, before, after in zip(change_times_pred, cases_before_pred, cases_after_pred):
            for ax in [axes[1]]:
                ax.axvline(x=t, color='green', linestyle='--', linewidth=1.5)
                ax.text(t, ax.get_ylim()[1]*0.98, f"{before}→{after}", color='green', rotation=90, va='top', ha='right')

        plt.tight_layout()
        fig.savefig(save_path + 'mixture_prediction_vs_orig-color.png')
        plt.close()

    if args.save_waves: 
        ## Using phase information
        phase_1 = np.angle(D1)
        phase_2 = np.angle(D2)
        phase_mix = np.angle(Dmix)

        # Original min and max values from STFT magnitudes, these should be known or have been computed previously
        orig_min_1, orig_max_1 = np.min(np.abs(D1)), np.max(np.abs(D1))
        orig_min_2, orig_max_2 = np.min(np.abs(D2)), np.max(np.abs(D2))
        orig_min_mix, orig_max_mix = np.min(np.abs(Dmix)), np.max(np.abs(Dmix))

        # No denormalization
        reconstructed_sp1 = full_mag_sp1 * np.exp(1j * phase_1)
        reconstructed_sp2 = full_mag_sp2 * np.exp(1j * phase_2)
        reconstructed_spmix = full_mag_mix * np.exp(1j * phase_mix)

        # Apply the inverse STFT to get the time-domain waveform
        reconstructed_sp1 = librosa.istft(reconstructed_sp1, hop_length=hop_length)
        reconstructed_sp2 = librosa.istft(reconstructed_sp2, hop_length=hop_length)
        reconstructed_mix = librosa.istft(reconstructed_spmix, hop_length=hop_length)

        sf.write(save_path + '1-reconstruction_speaker_1.wav', reconstructed_sp1, sr)
        sf.write(save_path + '1-reconstruction_speaker_2.wav', reconstructed_sp2, sr)
        sf.write(save_path + '1-reconstruction_speaker_mix.wav', reconstructed_mix, sr)

    if args.plot_supervised_semi_supervised:

        model_2 = MultiStream_VAE_audio(dimx=dimx, dimz=dimz, 
                                        n_sources=sources, device=device, 
                                        variational=variational, 
                                        freeze_decoder=freeze_decoder)
        
        model_2.load_state_dict(torch.load(args.model_path_2))
        pi_values_2 = [args.pi_1_2, args.pi_2_2]
        b_2 = args.b_2

        model_2.to(device)

        _, mode_vals_pred_2, full_mag_mix_2, _, _, time_intervals = run_inference(model_2)

        S6_dB = librosa.amplitude_to_db(full_mag_mix_2, ref=np.max)

        # Find where the case changes for target
        change_times_label = []
        cases_before_label = []
        cases_after_label = []

        for i in range(1, len(mode_vals_label)):
            if mode_vals_label[i] != mode_vals_label[i-1]:
                change_times_label.append(time_intervals[i][0])  # start time of interval where change occurs
                cases_before_label.append(mode_vals_label[i-1])
                cases_after_label.append(mode_vals_label[i])

        change_times_pred = []
        cases_before_pred = []
        cases_after_pred = []

        for i in range(1, len(mode_vals_pred)):
            if mode_vals_pred[i] != mode_vals_pred[i-1]:
                change_times_pred.append(time_intervals[i][0]) 
                cases_before_pred.append(mode_vals_pred[i-1])
                cases_after_pred.append(mode_vals_pred[i])

        change_times_pred_2 = []
        cases_before_pred_2 = []
        cases_after_pred_2 = []

        for i in range(1, len(mode_vals_pred_2)):
            if mode_vals_pred_2[i] != mode_vals_pred_2[i-1]:
                change_times_pred_2.append(time_intervals[i][0]) 
                cases_before_pred_2.append(mode_vals_pred_2[i-1])
                cases_after_pred_2.append(mode_vals_pred_2[i])
        
        # Regular plot in color
        fig, axes = plt.subplots(3, 1, figsize=(6.4, 6), sharex=True)

        librosa.display.specshow(Smix_dB, sr=fs, n_fft=512, hop_length=256, x_axis='time', y_axis='log', ax=axes[0])
        axes[0].set_title("Original Speaker Mixture")

        librosa.display.specshow(S5_dB, sr=fs, n_fft=512, hop_length=256, x_axis='time', y_axis='log', ax=axes[1])
        axes[1].set_title("Estimated Speaker Mixture (100% labels used in pretraining)")

        librosa.display.specshow(S6_dB, sr=fs, n_fft=512, hop_length=256, x_axis='time', y_axis='log', ax=axes[2])
        axes[2].set_title("Estimated Speaker Mixture (10% labels used in pretraining)")

        end_time = min_frames * hop_length / fs

        for ax in axes:
            ax.set_xlim(0, end_time)

        for t, before, after in zip(change_times_label, cases_before_label, cases_after_label):
            for ax in [axes[0]]:
                ax.axvline(x=t, color='red', linestyle='--', linewidth=1.5)
                ax.text(t, ax.get_ylim()[1]*0.98, f"{before}→{after}", color='red', rotation=90, va='top', ha='right')

        # Plot prediction change points in green on axes 1 and 3
        for t, before, after in zip(change_times_pred, cases_before_pred, cases_after_pred):
            for ax in [axes[1]]:
                ax.axvline(x=t, color='green', linestyle='--', linewidth=1.5)
                ax.text(t, ax.get_ylim()[1]*0.98, f"{before}→{after}", color='green', rotation=90, va='top', ha='right')

        for t, before, after in zip(change_times_pred_2, cases_before_pred_2, cases_after_pred_2):
            for ax in [axes[2]]:
                ax.axvline(x=t, color='green', linestyle='--', linewidth=1.5)
                ax.text(t, ax.get_ylim()[1]*0.98, f"{before}→{after}", color='green', rotation=90, va='top', ha='right')

        plt.tight_layout()
        fig.savefig(save_path + 'mixture_prediction_3_plots.png')
        plt.close()

    if args.compute_metrics_per_frame:

        test_path_s1 = args.test_dir_s1
        valid_files_s1 = sorted(glob.glob(test_path_s1 + '*.wav', recursive=True))

        test_path_s2 = args.test_dir_s2
        valid_files_s2 = sorted(glob.glob(test_path_s2 + '*.wav', recursive=True))

        test_dataset_mix = SpectrogramDatasetMetrics(valid_files_mix,
                                                        valid_files_s1,
                                                        valid_files_s2, 
                                                        winlen, 
                                                        transform=spec_transform, 
                                                        max_timesteps=17, 
                                                        transpose=False)


        # Initialize lists to hold components
        test_data_magnitude = []
        test_phase_mix = []
        test_wave_1 = []
        test_wave_2 = []

        scale = 0.0008  # Consistent scaling

        for idx in range(len(test_dataset_mix)):
            # Access data components from the dataset
            magnitude_norm_mix, phase_mix, wave_1, wave_2 = test_dataset_mix[idx]
            label = labels_test[idx]

            if label[1] == 0 and label[2] == 0:
                magnitude_norm_mix = torch.zeros_like(magnitude_norm_mix)  # Zero out based on label
                magnitude_norm_mix = Laplace(magnitude_norm_mix, scale).sample()

            if label[1] == 0:
                wave_1 = torch.zeros_like(wave_1)

            if label[2] == 0:
                wave_2 = torch.zeros_like(wave_2)

            # Append all components to their respective lists
            test_data_magnitude.append(magnitude_norm_mix)
            test_phase_mix.append(phase_mix)
            test_wave_1.append(wave_1)
            test_wave_2.append(wave_2)

        # Stack all components into tensors
        test_data_tensor = torch.stack(test_data_magnitude)
        test_phase_mix_tensor = torch.stack(test_phase_mix)
        test_wave_1_tensor = torch.stack(test_wave_1)
        test_wave_2_tensor = torch.stack(test_wave_2)

        labels_test = torch.Tensor(labels_test)  # Convert labels to tensor

        # Create tensor dataset with the required components
        test_set = TensorDataset(test_data_tensor, test_phase_mix_tensor, test_wave_1_tensor, test_wave_2_tensor, labels_test)

        # Create the dataloader with your configured test set
        spectrogram_dataloader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        average_metrics = run_inference_metrics(model)

    if args.compute_der:
        
        from pyannote.core import Annotation, Segment

        sp1_intervals = [
            ...
        ] # fill in all ground truth intervals where speaker 1 is active, in seconds, e.g., (19.0, 20.0), (97.0, 102.0)

        sp2_intervals = [
            ...
        ] # fill in all ground truth intervals where speaker 2 is active, in seconds, e.g., (19.0, 20.0), (97.0, 102.0)

        # Create Annotation object
        reference = Annotation()

        # Add speaker 1 segments
        for start, end in sp1_intervals:
            reference[Segment(start, end)] = "sp1"

        # Add speaker 2 segments
        for start, end in sp2_intervals:
            reference[Segment(start, end)] = "sp2"

        sp1_active = (q_s1_x_full > 0.99).astype(int)
        sp2_active = (q_s2_x_full > 0.99).astype(int)

        total_duration_sec = 120  # e.g., 2 minutes
        num_frames = 22501
        frame_duration = total_duration_sec / num_frames  # ≈ 0.00533 sec

        def mask_to_intervals(mask, frame_duration):
            intervals = []
            active = False
            start = None
            for i, val in enumerate(mask):
                if val == 1 and not active:
                    active = True
                    start = i * frame_duration
                elif val == 0 and active:
                    end = i * frame_duration
                    intervals.append((start, end))
                    active = False
            # Close any last interval
            if active:
                end = len(mask) * frame_duration
                intervals.append((start, end))
            return intervals

        sp1_intervals_pred = mask_to_intervals(sp1_active, frame_duration)
        sp2_intervals_pred = mask_to_intervals(sp2_active, frame_duration)

        def merge_intervals(intervals, max_gap=0.2, round_to=None):
            """
            intervals: list of (start, end), sorted by start (if not, sort them)
            max_gap: maximum gap between intervals to merge (in seconds)
            round_to: round boundaries to this value (e.g., 1 for whole seconds)
            """
            # Sort intervals by start time
            intervals = sorted(intervals, key=lambda x: x[0])
            merged = []
            
            for interval in intervals:
                start, end = interval
                if round_to is not None:
                    start = round(start / round_to) * round_to
                    end = round(end / round_to) * round_to
                if not merged:
                    merged.append((start, end))
                else:
                    prev_start, prev_end = merged[-1]
                    # If intervals overlap or are close enough, merge them
                    if start <= prev_end + max_gap:
                        merged[-1] = (prev_start, max(prev_end, end))
                    else:
                        merged.append((start, end))
            return merged

        # 
        sp1_intervals_pred_merged = merge_intervals(sp1_intervals_pred, max_gap=1, round_to=1)
        sp2_intervals_pred_merged = merge_intervals(sp2_intervals_pred, max_gap=1, round_to=1)
        
        hypothesis = Annotation()

        # Add speaker 1 segments
        for start, end in sp1_intervals_pred_merged:
            hypothesis[Segment(start, end)] = "sp1"

        # Add speaker 2 segments
        for start, end in sp2_intervals_pred_merged:
            hypothesis[Segment(start, end)] = "sp2"
        
        from pyannote.metrics.diarization import DiarizationErrorRate

        # Create DER metric object
        der = DiarizationErrorRate()

        # Evaluate
        details = der(reference, hypothesis, detailed=True)

        print("Miss%:    {:.2f}%".format(100 * details['missed detection'] / details['total']))
        print("Conf%:    {:.2f}%".format(100 * details['confusion'] / details['total']))
        print("FA%:      {:.2f}%".format(100 * details['false alarm'] / details['total']))
        print("Correct%: {:.2f}%".format(100 * details['correct'] / details['total']))
        print("DER:      {:.2f}%".format(100 * details['diarization error rate']))

        print('_')



