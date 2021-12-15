import abc

from omegaconf import DictConfig
from torch.nn import Module


class SequenceClassification(Module):
    def __init__(self, cfg: DictConfig):
        super(SequenceClassification, self).__init__()
        self.cfg = cfg
        self.encoder = self.instantiate_encoder()
        self.feature_transformer = self.instantiate_feature_transformer()
        self.classification_head = self.instantiate_classification_head()

    @abc.abstractmethod
    def forward(self, **kwargs):
        """
        steps:
        x = self.encoder(x)
        x = self.feature_transformer(x)
        x = self.classification_head(x)

        """
        raise NotImplementedError

    @abc.abstractmethod
    def instantiate_encoder(self):
        raise NotImplementedError

    @abc.abstractmethod
    def instantiate_feature_transformer(self):
        raise NotImplementedError

    @abc.abstractmethod
    def instantiate_classification_head(self):
        raise NotImplementedError
