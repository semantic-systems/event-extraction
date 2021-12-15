from omegaconf import DictConfig
from torch import tensor
from torch.nn import Linear, ModuleList, CrossEntropyLoss, Dropout
from transformers import AdamW

from models.heads import Head
from schema import EncodedFeature, SingleLabelClassificationForwardOutput, HeadOutput
import torch.nn.functional as F


class LinearLayerHead(Head):
    def __init__(self, cfg: DictConfig):
        super(LinearLayerHead, self).__init__()
        layer_stacks = [Linear(layer.n_in, layer.n_out) for layer in cfg.model.layers.values()]
        self.classification_layer = ModuleList(layer_stacks)
        self.dropout = Dropout(p=cfg.model.dropout_rate)

    def forward(self, encoded_features: EncodedFeature) -> HeadOutput:
        encoded_feature: tensor = encoded_features.encoded_feature
        output = encoded_feature
        for i, layer in enumerate(self.classification_layer):
            if i < len(self.classification_layer) - 1:
                if encoded_features.labels is None:
                    output = F.relu(layer(output))
                else:
                    output = F.relu(self.dropout(layer(output)))
            else:
                output = F.relu(layer(output))
        return HeadOutput(output)
