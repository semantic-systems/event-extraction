from typing import List, Dict
from abc import abstractmethod
from omegaconf import DictConfig
from torch.nn import Module, Linear, Dropout, ModuleList
from transformers import BertModel, AdamW, PreTrainedTokenizer
import torch.nn.functional as F
import torch


class SingleLabelSequenceClassification(Module):
    def __init__(self, cfg: DictConfig):
        super(SingleLabelSequenceClassification, self).__init__()
        self.cfg = cfg
        self.bert: PreTrainedTokenizer = BertModel.from_pretrained(cfg.from_pretrained)
        # TODO: overwrite the last layer n_out to the number of classes from the data loader.
        print(f"cfg.layers {cfg.layers} of type {type(cfg.layers)}")
        cfg_layers: DictConfig = cfg.layers
        self.classification_layers = self.get_layers(cfg_layers)
        self.optimizer = AdamW(self.parameters(), lr=cfg.learning_rate)
        self.dropout = Dropout(p=cfg.dropout_rate)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = output.pooler_output
        for i, layer in enumerate(self.classification_layers):
            if i < len(self.classification_layers) - 1:
                output = F.relu(layer(output))
                output = self.drop(output)
            else:
                output = F.softmax(layer(output))
        return output

    @staticmethod
    def get_layers(cfg_layers: DictConfig) -> ModuleList:
        layer_stacks: List = [Linear(layer.n_in, layer.n_out) for layer in cfg_layers.values()]
        return ModuleList(layer_stacks)

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


