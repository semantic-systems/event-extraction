import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceMaximizationCovarianceMinimizationLoss(nn.Module):
    # from the Yann LeCun VicReg paper
    # this loss maximizes the variance of each sample within the batch.
    def __init__(self, margin: float):
        super(VarianceMaximizationCovarianceMinimizationLoss, self).__init__()
        self.margin = margin

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x, y):
        batch_size = x.shape[0]
        num_features = x.shape[1]
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(num_features) + \
                   self.off_diagonal(cov_y).pow_(2).sum().div(num_features)
        return std_loss+ cov_loss