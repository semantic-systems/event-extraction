import pytest
from hydra import initialize, compose
from models.SingleLabelSequenceClassification import SingleLabelSequenceClassification
import pandas as pd


@pytest.fixture()
def hydra_config():
    with initialize(config_path="./configs", job_name="test"):
        # config is relative to a module
        cfg = compose(config_name="test_config_1.yaml")
    return cfg


@pytest.fixture()
def model_instance(hydra_config):
    model = SingleLabelSequenceClassification(hydra_config.model)
    return model


@pytest.fixture()
def mocked_df():
    sequences = ["Hallo ich bin zuhause."]
    df = pd.DataFrame({"sentence": sequences, "label": ["gold"]})
    return df
