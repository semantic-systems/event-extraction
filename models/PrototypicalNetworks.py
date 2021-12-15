import torch
from omegaconf import DictConfig
from torch import tensor
from torch.nn import CrossEntropyLoss, Dropout, Identity
from transformers import PreTrainedModel, AdamW, PreTrainedTokenizer, BertTokenizer, AutoModel

from models import SequenceClassification
from models.heads.prototypical_head import PrototypicalHead
from schema import InputFeature


class PrototypicalNetworks(SequenceClassification):
    def __init__(self, cfg: DictConfig):
        super(PrototypicalNetworks, self).__init__(cfg)
        self.encoder: PreTrainedModel = self.encoder_layer(cfg)
        self.classification_layers = self.classification_head(cfg)
        self.optimizer = AdamW(self.parameters(), lr=cfg.model.learning_rate)
        self.loss = CrossEntropyLoss()
        self.dropout = Dropout(p=cfg.model.dropout_rate)
        self.tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained(cfg.model.from_pretrained)

    def forward( self, support_features: InputFeature, query_features: InputFeature) -> tensor:
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

    def classification_head(self, config: DictConfig):
        return PrototypicalHead()

    def instantiate_encoder(self) -> PreTrainedModel:
        encoder: PreTrainedModel = AutoModel.from_pretrained(self.cfg.model.from_pretrained)
        encoder = self.trim_encoder_layers(encoder, self.cfg.model.num_transformer_layers)
        return encoder

    def instantiate_feature_transformer(self):
        # this returns an empty Module
        return Identity()

    def instantiate_classification_head(self):
        return PrototypicalHead()
