# Copyright (C) 2026 Veranika Boukun (veranika.boukun@uni-oldenburg.de), Jörg Lücke

from torch.utils.data import Dataset


class ImageOnlyDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        img, _ = self.subset[index] 
        return img

    def __len__(self):
        return len(self.subset)