import abc

import pandas as pd
from torch.utils.data import Dataset
import torch
from torch import tensor


class CustomDataset(Dataset):
    # implementation of custom dataset following huggingface's structure
    # https://huggingface.co/transformers/custom_datasets.html
    def __init__(self, df: pd.DataFrame, sentence_column: str, label_column: str, from_pretrained: str):
        self.df = df
        self.sentence_column = sentence_column
        self.label_column = label_column
        self.from_pretrained = from_pretrained
        self.encodings: tensor = self.preprocess()

    @abc.abstractmethod
    def preprocess(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.df)

    # @property
    # @abc.abstractmethod
    # def tokenizer(self):
    #     raise NotImplementedError

