from distutils import dist

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceMaximizationCovarianceMinimizationLoss(nn.Module):
    # from the Yann LeCun VicReg paper
    # this loss maximizes the variance of each sample within the batch,
    # and minimizes the covariance of the batch.
    # the goal is to push the energy up / free-energy minimization.
    def __init__(self, margin: float):
        super(VarianceMaximizationCovarianceMinimizationLoss, self).__init__()
        self.margin = margin

    @staticmethod
    def off_diagonal(x):
        if x.shape == torch.Size([]):
            return torch.tensor([1e-4])
        else:
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    @staticmethod
    def construct_multiview_features_per_class(features, labels):
        """
        construct multiview features per class as a dict
        :param features: batch_size * num_views * num_features
        :param labels: batch_size
        :return:
        """
        classes = torch.unique(labels)
        chunks = {c.item(): 0 for c in classes}
        for c in classes:
            indices = (labels == c).nonzero(as_tuple=True)[0]
            chunks[c.item()] = torch.cat(torch.index_select(features[:, :], 0, indices).unbind(0))
        return chunks

    def std_loss(self, multiview_features_per_class, device):
        classes = list(multiview_features_per_class.keys())
        loss = 0
        for c in classes:
            normalized_features_per_class = multiview_features_per_class[c] - multiview_features_per_class[c].mean(dim=0)
            std_features_per_class = torch.sqrt(normalized_features_per_class.var(dim=0) + 0.0001)
            loss += torch.mean(F.relu(1 - std_features_per_class))
        return loss

    def cov_loss(self, multiview_features_per_class, device):
        classes = list(multiview_features_per_class.keys())
        loss = 0
        for c in classes:
            normalized_features_per_class = multiview_features_per_class[c] - multiview_features_per_class[c].mean(dim=0)
            size = normalized_features_per_class.shape[0]
            cov = (normalized_features_per_class.T @ normalized_features_per_class) / (size - 1)
            loss += self.off_diagonal(cov).pow_(2).sum().div(normalized_features_per_class.shape[-1])
        return loss

    def forward(self, features, labels):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        multiview_features_per_class = self.construct_multiview_features_per_class(features, labels)
        return self.std_loss(multiview_features_per_class, device) + self.cov_loss(multiview_features_per_class, device)
