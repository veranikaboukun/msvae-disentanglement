# Copyright (C) 2026 Veranika Boukun (veranika.boukun@uni-oldenburg.de), Jörg Lücke

from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import numpy as np


def get_subset_dataloaders_and_indices(directory, batch_size, subset, percent_labeled, save_path, kwargs):
    train_set = datasets.MNIST(directory, train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST(directory, train=False, download=True, transform=transforms.ToTensor())

    indices_train = [i for i, (img, label) in enumerate(train_set) if label in subset]

    np.random.seed(42)
    np.random.shuffle(indices_train)
    split_idx = int(len(indices_train) * percent_labeled)
    indices_pretraining = indices_train[:split_idx]
    indices_training = indices_train[split_idx:]

    np.save(save_path+'indices_pretraining.npy', indices_pretraining)
    np.save(save_path+'indices_training.npy', indices_training)

    indices_test = [i for i, (img, label) in enumerate(test_set) if label in subset]

    np.random.shuffle(indices_test)
    split_idx = int(len(indices_test) * percent_labeled)
    indices_pretesting = indices_test[:split_idx]
    indices_testing = indices_test[split_idx:]

    np.save(save_path+'indices_pretesting.npy', indices_pretesting)
    np.save(save_path+'indices_testing.npy', indices_testing)
    
    subset_train = Subset(train_set, indices_pretraining)
    subset_test = Subset(test_set, indices_pretesting)

    train_loader = DataLoader(subset_train, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(subset_test, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader
