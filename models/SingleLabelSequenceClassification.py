import copy
from typing import List, Optional
from abc import abstractmethod
from omegaconf import DictConfig
from torch.nn import Module, Linear, Dropout, ModuleList
from torch import tensor
from transformers import AutoModel, AdamW, PreTrainedTokenizer, PreTrainedModel, BertTokenizer
import torch.nn.functional as F
import torch


class SingleLabelSequenceClassification(Module):
    def __init__(self, cfg: DictConfig):
        super(SingleLabelSequenceClassification, self).__init__()
        self.cfg = cfg
        # self.tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained(cfg.from_pretrained)
        self.bert: PreTrainedModel = AutoModel.from_pretrained(cfg.from_pretrained)
        self.bert = self.trim_encoder_layers(cfg.n_layers)
        # TODO: overwrite the last layer n_out to the number of classes from the data loader.
        cfg_layers: DictConfig = cfg.layers
        self.classification_layers = self.get_layers(cfg_layers)
        self.optimizer = AdamW(self.parameters(), lr=cfg.learning_rate)
        self.dropout = Dropout(p=cfg.dropout_rate)

    def forward(self, input_ids: tensor, attention_mask: tensor, labels: Optional[tensor] = None):
        if labels is None:
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            for i, layer in enumerate(self.classification_layers):
                if i < len(self.classification_layers) - 1:
                    output = F.relu(layer(output))
                    output = self.dropout(output)
                else:
                    output = F.softmax(layer(output))
            return output

    @staticmethod
    def get_layers(cfg_layers: DictConfig) -> ModuleList:
        layer_stacks: List = [Linear(layer.n_in, layer.n_out) for layer in cfg_layers.values()]
        return ModuleList(layer_stacks)

    def trim_encoder_layers(self, num_layers_to_keep: int) -> PreTrainedModel:
        full_layers = self.bert.encoder.layer
        trimmed_layers = ModuleList()

        # Now iterate over all layers, only keeping only the relevant layers.
        for i in range(num_layers_to_keep):
            trimmed_layers.append(full_layers[i])

        # create a copy of the model, modify it with the new list, and return
        trimmed_encoder = copy.deepcopy(self.bert)
        trimmed_encoder.encoder.layer = trimmed_layers
        return trimmed_encoder

    @abstractmethod
    def train_model(self):
        raise NotImplemented

    @abstractmethod
    def test_model(self):
        raise NotImplemented

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")


