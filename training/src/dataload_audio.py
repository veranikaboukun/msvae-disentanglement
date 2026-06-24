# Copyright (C) 2026 Veranika Boukun (veranika.boukun@uni-oldenburg.de), Jörg Lücke

import os
import random
random.seed(123)
import numpy as np
np.random.seed(123)
import torch
from torch.utils import data
from torchaudio.transforms import Spectrogram
import torchaudio
import torch.nn.functional as F

class SpectrogramDataset(data.Dataset):
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.files = os.listdir(file_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = os.path.join(self.file_dir, self.files[idx])

        sample = np.load(file)
        sample_normalized = (sample - sample.min())/(sample.max()-sample.min())
        sample_transposed = sample_normalized.T

        return torch.from_numpy(sample_transposed).float() 

class SpeechSpectrogramDataset(data.Dataset):
    def __init__(self, filenames, win_length, 
                 transform=None, target_length=None, max_timesteps=None, 
                 transpose=True):
        self.filenames = filenames
        self.win_length = win_length
        self.transform = transform or Spectrogram(win_length=win_length, 
                                                  n_fft=win_length,
                                                  hop_length=win_length//2)
        self.target_length = target_length

        self.max_timesteps = max_timesteps 
        self.transpose = transpose

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_path = self.filenames[idx]
        waveform, sr = torchaudio.load(audio_path)       
        yt = waveform[0]

        spectrogram = self.transform(yt)

        # # Zero-pad along the time dimension
        spectrogram = F.pad(input=spectrogram, 
                            pad=(0, self.max_timesteps - spectrogram.shape[1]),
                            mode='constant', 
                            value=0)
        
        min_val = spectrogram.min()
        max_val = spectrogram.max()
        eps = 1e-6 
        spectrogram_normalized = (spectrogram - min_val) / max((max_val - min_val), eps)

        if self.transpose:
            spectrogram_normalized = spectrogram_normalized.T

        else:
            spectrogram_normalized = spectrogram_normalized

        return spectrogram_normalized 
    