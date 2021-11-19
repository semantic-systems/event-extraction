from pathlib import Path
from abc import abstractmethod
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


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
