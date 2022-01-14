import torch
import logging
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from typing import List, Optional, Tuple
from pathlib import Path
from torch import tensor
from torch.nn import CrossEntropyLoss, Dropout, Identity
from tqdm import tqdm
from transformers import PreTrainedModel, AdamW, PreTrainedTokenizer, BertTokenizer, AutoModel
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay

from models import SequenceClassification
from models.heads.prototypical_head import PrototypicalHead
from schema import InputFeature
from helper import log_metrics, set_run_testing, set_run_training, get_data_time
from schema import SingleLabelClassificationForwardOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PrototypicalNetworks(SequenceClassification):
    def __init__(self, cfg: DictConfig):
        super(PrototypicalNetworks, self).__init__(cfg)
        self.encoder: PreTrainedModel = self.encoder_layer(cfg)
        self.classification_layers = self.classification_head(cfg)
        self.optimizer = AdamW(self.parameters(), lr=cfg.model.learning_rate)
        self.loss = CrossEntropyLoss()
        self.dropout = Dropout(p=cfg.model.dropout_rate)
        self.tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained(cfg.model.from_pretrained)

    def forward(self, support_features: InputFeature, query_features: InputFeature) -> tensor:
        support_labels: tensor = support_features.labels
        support_ids: tensor = support_features.input_ids
        support_attention_mask: tensor = support_features.attention_mask
        query_ids: tensor = query_features.input_ids
        query_attention_mask: tensor = query_features.attention_mask

        # Infer the number of classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all support features vector with label i
        support_encoded_feature = self.encoder(input_ids=support_ids, attention_mask=support_attention_mask).pooler_output
        query_encoded_feature = self.encoder(input_ids=query_ids, attention_mask=query_attention_mask).pooler_output

        z_proto = torch.cat(
            [
                support_encoded_feature[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(query_encoded_feature, z_proto)

        scores = -dists
        return scores

    def instantiate_classification_head(self):
        return PrototypicalHead(self.cfg)

    def instantiate_encoder(self) -> PreTrainedModel:
        encoder: PreTrainedModel = AutoModel.from_pretrained(self.cfg.model.from_pretrained)
        encoder = self.trim_encoder_layers(encoder, self.cfg.model.num_transformer_layers)
        return encoder

    def instantiate_feature_transformer(self):
        # this returns an empty Module
        return Identity()

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