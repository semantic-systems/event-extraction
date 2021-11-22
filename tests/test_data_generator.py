from data_generators.data_generator import DataGenerator
import pandas as pd


def test_initialize_data_generator(hydra_config):
    cfg = hydra_config
    generator = DataGenerator(cfg.data)
    assert generator
    assert isinstance(generator.testing_data, pd.DataFrame)
    assert isinstance(generator.training_data, pd.DataFrame)
