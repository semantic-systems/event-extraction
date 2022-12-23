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
        chunks = {v: {c.item(): 0 for c in classes} for v in range(features.shape[1])}
        for v in range(features.shape[1]):
            for c in classes:
                indices = (labels == c).nonzero(as_tuple=True)[0]
                chunks[v][c.item()] = torch.index_select(features[:, v], 0, indices)
        return chunks

    def std_loss(self, multiview_features_per_class):
        views = list(multiview_features_per_class.keys())
        classes = list(multiview_features_per_class[0].keys())
        loss = 0
        for view in views:
            for c in classes:
                std_per_class_in_view = torch.var(multiview_features_per_class[view][c], unbiased=True)
                loss += torch.mean(F.relu(1 - std_per_class_in_view)) / (len(views) * len(classes))
        return loss

    def cov_loss(self, multiview_features_per_class):
        views = list(multiview_features_per_class.keys())
        classes = list(multiview_features_per_class[0].keys())
        loss = 0
        for view in views:
            for c in classes:
                cov_per_class_in_view = torch.cov(multiview_features_per_class[view][c])
                loss += self.off_diagonal(cov_per_class_in_view).pow_(2).sum().div(
                    multiview_features_per_class[view][c].shape[-1])
        return loss

    def forward(self, features, labels):
        multiview_features_per_class = self.construct_multiview_features_per_class(features, labels)
        return self.std_loss(multiview_features_per_class) + self.cov_loss(multiview_features_per_class)