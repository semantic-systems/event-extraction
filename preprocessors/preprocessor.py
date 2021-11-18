from abc import abstractmethod
import pandas as pd


class Preprocessor(object):
    def __init__(self, df: pd.Dataframe):
        self.df = df

    @abstractmethod
    def preprocess(self):
        raise NotImplemented