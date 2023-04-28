import torch
from torch.utils.data import Dataset


class CountingDataset(Dataset):
    def __init__(self, matrices, labels):
        self.matrices = matrices
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        label = self.labels[idx]

        # Convert matrix to PyTorch tensor
        matrix = torch.tensor(matrix, dtype=torch.float).reshape((1, 28, 28))

        # Convert label to PyTorch tensor
        label = torch.tensor(label, dtype=torch.long)

        return matrix, label
