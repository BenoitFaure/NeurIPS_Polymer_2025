import torch
import numpy as np

class LabelNormalizer:
    def __init__(self, means : np.ndarray, stds: np.ndarray):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

    def normalize(self, labels) -> torch.Tensor:
        return (labels - self.means) / self.stds

    def denormalize(self, labels) -> torch.Tensor:
        return labels * self.stds + self.means