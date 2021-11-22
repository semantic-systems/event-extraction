from pathlib import Path
from abc import abstractmethod
from typing import Optional

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler, RandomSampler


class Preprocessor(object):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.training_data = self.load_csv(Path(cfg.training_data_path).absolute())
        if cfg.testing_data_path:
            self.testing_data = self.load_csv(Path(cfg.testing_data_path).absolute())
        else:
            self.training_data, self.testing_data = train_test_split(
                self.training_data, train_size=cfg.train_test_split)

    @abstractmethod
    def preprocess(self):
        raise NotImplemented

    @staticmethod
    def load_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    def __call__(self,
                 dataset: Dataset,
                 batch_size: int,
                 shuffle: Optional[bool] = False,
                 sampler: Optional[Sampler] = RandomSampler) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

