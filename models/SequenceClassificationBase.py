import abc
import copy
import torch
from typing import Union
from omegaconf import DictConfig
from torch.nn import Module, ModuleList
from transformers import PreTrainedModel


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

    @staticmethod
    def trim_encoder_layers(encoder: PreTrainedModel, num_layers_to_keep: int) -> PreTrainedModel:
        full_layers = encoder.encoder.layer
        trimmed_layers = ModuleList()

        # Now iterate over all layers, only keeping only the relevant layers.
        for i in range(num_layers_to_keep):
            trimmed_layers.append(full_layers[i])

        # create a copy of the model, modify it with the new list, and return
        trimmed_encoder = copy.deepcopy(encoder)
        trimmed_encoder.encoder.layer = trimmed_layers
        return trimmed_encoder

    @staticmethod
    def freeze_encoder(encoder: PreTrainedModel, layers_to_freeze: Union[str, int]) -> PreTrainedModel:
        if layers_to_freeze == "none":
            return encoder
        elif layers_to_freeze == "all":
            for param in encoder.parameters():
                param.requires_grad = False
            return encoder
        elif isinstance(layers_to_freeze, int) and layers_to_freeze <= len(encoder.encoder.layer) - 1:
            modules = [encoder.embeddings, *encoder.encoder.layer[:layers_to_freeze]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
            return encoder
        else:
            raise ValueError(f"Currently, only 'all', 'none' and integer (<=num_transformer_layers) "
                             f"are valid value for freeze_transformer_layer")

    def preprocess(self, batch):
        return self.tokenizer(batch["text"], padding=True, truncation=True, max_length=512, return_tensors="pt")

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")