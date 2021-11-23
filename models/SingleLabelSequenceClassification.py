import copy
import logging

from typing import List, Optional

import mlflow
from omegaconf import DictConfig
from sklearn.metrics import ConfusionMatrixDisplay
from torch.nn import Module, Linear, Dropout, ModuleList, CrossEntropyLoss
from torch import tensor
from torch.utils.data import DataLoader
from transformers import AutoModel, AdamW, PreTrainedModel
from schema import SingleLabelClassificationForwardOutput
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from utils import log_params_from_omegaconf_dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SingleLabelSequenceClassification(Module):
    def __init__(self, cfg: DictConfig, num_classes: int):
        super(SingleLabelSequenceClassification, self).__init__()
        self.cfg = cfg
        self.bert: PreTrainedModel = AutoModel.from_pretrained(cfg.model.from_pretrained)
        self.bert = self.trim_encoder_layers(cfg.model.num_transformer_layers)
        self.cfg.model.layers = self.fill_config_with_num_classes(cfg.model.layers, num_classes)
        self.classification_layers = self.get_layers(self.cfg.model.layers)
        self.optimizer = AdamW(self.parameters(), lr=cfg.model.learning_rate)
        self.loss = CrossEntropyLoss()
        self.dropout = Dropout(p=cfg.model.dropout_rate)

    def forward(self,
                input_ids: tensor,
                attention_mask: tensor,
                labels: Optional[tensor] = None) -> SingleLabelClassificationForwardOutput:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        for i, layer in enumerate(self.classification_layers):
            if i < len(self.classification_layers) - 1:
                output = F.relu(layer(output))
                if labels is None:
                    output = self.dropout(output)
            else:
                output = F.softmax(layer(output))
        if labels is not None:
            loss = self.loss(output, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return SingleLabelClassificationForwardOutput(loss=loss, prediction_logits=output)
        else:
            return SingleLabelClassificationForwardOutput(prediction_logits=output)

    @staticmethod
    def get_layers(cfg_layers: DictConfig) -> ModuleList:
        layer_stacks: List = [Linear(layer.n_in, layer.n_out) for layer in cfg_layers.values()]
        return ModuleList(layer_stacks)

    @staticmethod
    def fill_config_with_num_classes(cfg_layer: DictConfig, num_classes: int) -> DictConfig:
        updated_config = copy.deepcopy(cfg_layer)
        for n, (key, value) in enumerate(list(cfg_layer.items())):
            if n == len(list(cfg_layer.values())) - 1:
                updated_config[key]["n_out"] = num_classes
        return updated_config

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

    def train_model(self, data_loader: DataLoader):
        self.to(self.device)
        self.train()
        self.optimizer.zero_grad()
        y_true = []
        y_predict = []
        # TODO: do we want to configure the uri?
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        with mlflow.start_run():
            mlflow.set_experiment(self.cfg.name+"_training")
            log_params_from_omegaconf_dict(self.cfg)
            # start new run
            for n in range(self.cfg.model.epochs):
                for batch in data_loader:
                    input_ids: tensor = batch["input_ids"].to(self.device)
                    attention_masks: tensor = batch["attention_mask"].to(self.device)
                    labels: tensor = batch["labels"].to(self.device)
                    y_true.extend(labels.tolist())
                    outputs: SingleLabelClassificationForwardOutput = self(input_ids, attention_masks, labels)
                    loss = outputs.loss
                    prediction = outputs.prediction_logits.max(1).indices.to("cpu")
                    y_predict.extend(prediction.tolist())
                acc = (prediction == labels).sum().item() / prediction.size(0)
                # log metric
                mlflow.log_metric("loss", loss.item(), step=1)
                mlflow.log_metric("train_acc", acc, step=1)
                # if n % 1 == 0:
                logger.warning(f"Epoch: {n}, Average loss: {loss.item()}")
                ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_predict)
                plt.savefig(f"./outputs/confusion_matrix_train_epoch_{n}.png")
                mlflow.log_artifact(f"./outputs/confusion_matrix_train_epoch_{n}.png")
                plt.close()
        torch.save(self, "./outputs/test_model.pt")

    def test_model(self, data_loader: DataLoader):
        self.eval()
        y_true = []
        y_predict = []
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        with mlflow.start_run():
            mlflow.set_experiment(self.cfg.name+"_testing")
            log_params_from_omegaconf_dict(self.cfg)
            with torch.no_grad():
                for batch in data_loader:
                    input_ids: tensor = batch["input_ids"].to(self.device)
                    attention_masks: tensor = batch["attention_mask"].to(self.device)
                    labels: tensor = batch["labels"]
                    outputs: SingleLabelClassificationForwardOutput = self(input_ids, attention_masks)
                    y_true.extend(labels.tolist())
                    prediction = outputs.prediction_logits.max(1).indices
                    y_predict.extend(prediction.tolist())
                acc = (prediction == labels).sum().item() / prediction.size(0)
                mlflow.log_metric("test_acc", acc, step=1)
                ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_predict)
                plt.savefig("./outputs/confusion_matrix_test.png")
                mlflow.log_artifact("./outputs/confusion_matrix_test.png")
                plt.close()

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
