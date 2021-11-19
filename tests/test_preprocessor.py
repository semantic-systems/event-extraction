from hydra import compose
from preprocessors.preprocessor import Preprocessor
import pandas as pd


def test_initialize_model(initialize_hydra):
    cfg = compose(config_name="test_config_1.yml")
    preprocessor = Preprocessor(cfg.data)
    assert preprocessor
    assert isinstance(preprocessor.testing_data, pd.DataFrame)
    assert isinstance(preprocessor.training_data, pd.DataFrame)
