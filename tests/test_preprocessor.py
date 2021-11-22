from hydra import compose
from preprocessors.preprocessor import Preprocessor
import pandas as pd


def test_initialize_preprocessor(hydra_config):
    cfg = hydra_config
    preprocessor = Preprocessor(cfg.data)
    assert preprocessor
    assert isinstance(preprocessor.testing_data, pd.DataFrame)
    assert isinstance(preprocessor.training_data, pd.DataFrame)
