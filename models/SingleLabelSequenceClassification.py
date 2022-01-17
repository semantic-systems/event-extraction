import copy
import logging
from itertools import chain
from pathlib import Path

from typing import List, Optional, Tuple

from omegaconf import DictConfig
from sklearn.metrics import ConfusionMatrixDisplay
from torch.nn import Module, CrossEntropyLoss, Identity
from torch import tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AdamW, PreTrainedModel, PreTrainedTokenizer, BertTokenizer

from helper import log_metrics, set_run_testing, set_run_training, get_data_time
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

    def forward(self, input_feature: InputFeature) -> SingleLabelClassificationForwardOutput:
        output = self.encoder(input_ids=input_feature.input_ids, attention_mask=input_feature.attention_mask).pooler_output
        encoded_feature: EncodedFeature = EncodedFeature(encoded_feature=output, labels=input_feature.labels)
        head_output = self.classification_head(encoded_feature)

        if input_feature.labels is not None:
            loss = self.loss(head_output.output, input_feature.labels)
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

    @set_run_training
    def train_model(self, data_loader: DataLoader):
        self.to(self.device)
        self.train()
        self.optimizer.zero_grad()
        # start new run
        for n in tqdm(range(self.cfg.model.epochs)):
            y_predict, y_true, loss = self.run_per_epoch(data_loader)
            acc, _, _ = self.evaluate(y_predict, y_true, loss, num_epoch=n)
            logger.warning(f"Epoch: {n}, Average loss: {loss}, Average acc: {acc}")
        torch.save(self, Path(self.cfg.model.output_path, self.cfg.name, "pretrained_models",
                              f"{self.cfg.name}_{get_data_time()}.pt").absolute())

    def run_per_epoch(self, data_loader: DataLoader, test: Optional[bool] = False) -> Tuple[List, List, int]:
        y_predict, y_true = [], []
        loss = 0
        for i, batch in enumerate(data_loader):
            if not test:
                self.optimizer.zero_grad()
            labels: tensor = batch["label"].to(self.device)
            y_true.extend(labels)
            batch = self.preprocess(batch)
            input_ids: tensor = batch["input_ids"].to(self.device)
            attention_masks: tensor = batch["attention_mask"].to(self.device)
            # convert labels to None if in testing mode.
            labels = None if test else labels
            input_feature: InputFeature = InputFeature(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
            outputs: SingleLabelClassificationForwardOutput = self(input_feature)
            prediction = outputs.prediction_logits.argmax(1)
            y_predict.extend(prediction)
            if not test:
                loss = (loss + outputs.loss.item())/(i+1)
        return y_predict, y_true, loss

    @log_metrics
    def evaluate(self,
                 y_predict: List,
                 y_true: List,
                 loss: int,
                 num_epoch: Optional[int] = None) -> Tuple[float, float, str]:
        y_predict = torch.stack(y_predict)
        y_true = torch.stack(y_true)
        acc = (y_predict == y_true).sum().item() / y_predict.size(0)
        ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_predict)
        if num_epoch is not None:
            path_to_plot: str = str(Path(self.cfg.model.output_path, self.cfg.name, "plots",
                                         f'confusion_matrix_train_epoch_{num_epoch}.png').absolute())
        else:
            path_to_plot = str(Path(self.cfg.model.output_path, self.cfg.name, "plots",
                                    'confusion_matrix_test.png').absolute())
        plt.savefig(path_to_plot)
        plt.close()
        return acc, loss, path_to_plot

    @set_run_testing
    def test_model(self, data_loader: DataLoader):
        self.eval()
        with torch.no_grad():
            y_predict, y_true, loss = self.run_per_epoch(data_loader, test=True)
            acc, _, _ = self.evaluate(y_predict, y_true, loss)
            logger.warning(f"Testing Accuracy: {acc}")

