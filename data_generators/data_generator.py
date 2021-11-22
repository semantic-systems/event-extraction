import abc
from pathlib import Path
from typing import Optional

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Sampler, RandomSampler


class DataGenerator(object):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.training_dataframe = self.load_csv(Path(cfg.data.training_data_path).absolute())
        if cfg.data.testing_data_path:
            self.testing_dataframe = self.load_csv(Path(cfg.data.testing_data_path).absolute())
        else:
            self.training_dataframe, self.testing_dataframe = train_test_split(
                self.training_dataframe, train_size=cfg.data.train_test_split)

    @property
    @abc.abstractmethod
    def training_dataset(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def testing_dataset(self):
        raise NotImplementedError

    @staticmethod
    def load_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    def __call__(self,
                 mode: str,
                 batch_size: int,
                 shuffle: Optional[bool] = False,
                 sampler: Optional[Sampler] = RandomSampler) -> DataLoader:
        """
        :param mode: train, valid or test
        :param batch_size:
        :param shuffle:
        :param sampler:
        :return:
        """
        if mode == "train":
            dataset = self.training_dataset
        elif mode == "test":
            dataset = self.testing_dataset
        else:
            raise AttributeError(f"{mode} is not a valid attribute in Data Generator class.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

