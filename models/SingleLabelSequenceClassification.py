import abc
import copy
import logging
from itertools import chain

from typing import List, Optional, Union

import mlflow
from omegaconf import DictConfig
from sklearn.metrics import ConfusionMatrixDisplay
from torch.nn import Module, Linear, Dropout, ModuleList, CrossEntropyLoss, Identity
from torch import tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AdamW, PreTrainedModel, PreTrainedTokenizer, BertTokenizer

from helper import set_run, log_metrics
from models import SequenceClassification
from models.heads import LinearLayerHead
from schema import SingleLabelClassificationForwardOutput, InputFeature, EncodedFeature
import torch
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SingleLabelSequenceClassification(SequenceClassification):
    def __init__(self, cfg: DictConfig):
        super(SingleLabelSequenceClassification, self).__init__(cfg)
        self.tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained(cfg.model.from_pretrained)
        params = chain(self.encoder.parameters(), self.classification_head.parameters())
        self.optimizer = AdamW(params, lr=cfg.model.learning_rate)
        self.loss = CrossEntropyLoss()

    def forward(self,
                input_feature: InputFeature,
                labels: Optional[tensor] = None) -> SingleLabelClassificationForwardOutput:
        output = self.encoder(input_ids=input_feature.input_ids, attention_mask=input_feature.attention_mask).pooler_output
        encoded_feature: EncodedFeature = EncodedFeature(encoded_feature=output, labels=labels)
        head_output = self.classification_head(encoded_feature)

        if labels is not None:
            loss = self.loss(head_output.output, labels)
            loss.backward()
            self.optimizer.step()
            return SingleLabelClassificationForwardOutput(loss=loss, prediction_logits=head_output.output)
        else:
            return SingleLabelClassificationForwardOutput(prediction_logits=head_output.output)

    def instantiate_encoder(self) -> PreTrainedModel:
        encoder: PreTrainedModel = AutoModel.from_pretrained(self.cfg.model.from_pretrained)
        encoder = self.trim_encoder_layers(encoder, self.cfg.model.num_transformer_layers)
        encoder = self.freeze_encoder(encoder, self.cfg.model.freeze_transformer_layers)
        return encoder

    def instantiate_feature_transformer(self) -> Module:
        # this returns an empty Module
        return Identity()

    def instantiate_classification_head(self) -> LinearLayerHead:
        return LinearLayerHead(self.cfg)

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

    def preprocess(self, batch):
        return self.tokenizer(batch["text"], padding=True, truncation=True, max_length=512, return_tensors="pt")

    @set_run
    def train_model(self, data_loader: DataLoader):
        torch.manual_seed(42)
        self.to(self.device)
        self.train()
        self.optimizer.zero_grad()
        # start new run
        for n in tqdm(range(self.cfg.model.epochs)):
            loss, acc = self.run_per_epoch(data_loader, n)
            logger.warning(f"Epoch: {n}, Average loss: {loss.item()}, Average acc: {acc}")
        torch.save(self, f"./outputs/test_model_{self.cfg.name}.pt")

    @log_metrics
    def run_per_epoch(self, data_loader: DataLoader, num_epoch: str, test: Optional[bool] = False):
        labels, loss, prediction = None, None, None
        y_predict, y_true = [], []
        for i, batch in enumerate(data_loader):
            self.optimizer.zero_grad()
            labels: tensor = batch["label"].to(self.device)
            batch = self.preprocess(batch)
            input_ids: tensor = batch["input_ids"].to(self.device)
            attention_masks: tensor = batch["attention_mask"].to(self.device)
            input_feature: InputFeature = InputFeature(input_ids=input_ids, attention_mask=attention_masks)
            y_true.extend(labels)
            labels = None if test else labels
            outputs: SingleLabelClassificationForwardOutput = self(input_feature, labels=labels)
            loss = outputs.loss
            prediction = outputs.prediction_logits.argmax(1)
            y_predict.extend(prediction)
        assert (labels is not None) and (loss is not None) and (prediction is not None)
        y_predict = torch.stack(y_predict)
        y_true = torch.stack(y_true)
        acc = (y_predict == y_true).sum().item() / y_predict.size(0)
        ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_predict)
        plt.savefig(f"./outputs/confusion_matrix_train_epoch_{num_epoch}.png")
        plt.close()
        return loss, acc

    # @set_run
    def test_model(self, data_loader: DataLoader):
        torch.manual_seed(42)
        self.eval()
        y_true = []
        y_predict = []
        with torch.no_grad():
            for batch in data_loader:
                labels: tensor = batch["label"].to(self.device)
                batch = self.preprocess(batch)
                input_ids: tensor = batch["input_ids"].to(self.device)
                attention_masks: tensor = batch["attention_mask"].to(self.device)
                input_feature: InputFeature = InputFeature(input_ids=input_ids, attention_mask=attention_masks)
                outputs: SingleLabelClassificationForwardOutput = self(input_feature)
                y_true.extend(labels)
                prediction = outputs.prediction_logits.argmax(1)
                y_predict.extend(prediction)
            y_predict = torch.stack(y_predict)
            y_true = torch.stack(y_true)
            acc = (y_predict == y_true).sum().item() / y_predict.size(0)
            mlflow.log_metric("test_acc", acc, step=1)
            ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_predict)
            plt.savefig("./outputs/confusion_matrix_test.png")
            mlflow.log_artifact("./outputs/confusion_matrix_test.png")
            plt.close()
            logger.warning(f"Test accuracy: {acc}")

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
