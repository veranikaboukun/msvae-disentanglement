from torch.utils.data import Dataset


class ImageOnlyDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        img, _ = self.subset[index]  # Ignore label
        return img

    def __len__(self):
        return len(self.subset)