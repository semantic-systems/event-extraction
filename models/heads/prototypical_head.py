import torch
from omegaconf import DictConfig
from torch import tensor

from models.heads import Head
from schema import EncodedFeature, PrototypicalNetworksForwardOutput


class PrototypicalHead(Head):
    def __init__(self, cfg: DictConfig):
        super(PrototypicalHead, self).__init__()
        self.cfg = cfg

    def forward(
        self,
        support_features: EncodedFeature,
        query_features: EncodedFeature,
    ) -> PrototypicalNetworksForwardOutput:

        support_labels: tensor = support_features.labels
        support_feature: tensor = support_features.encoded_feature
        query_feature: tensor = query_features.encoded_feature

        # Infer the number of classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all support features vector with label i

        z_proto = torch.cat(
            [
                support_feature[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(query_feature, z_proto)

        scores = -dists
        return PrototypicalNetworksForwardOutput(dists)
