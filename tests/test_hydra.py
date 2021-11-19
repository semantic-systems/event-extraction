from hydra import compose
from omegaconf import DictConfig


def test_nested_dict_in_hydra_is_iterable(initialize_hydra):
    cfg = compose(config_name="test_config_1.yml")
    assert isinstance(cfg, DictConfig)
    layers = [(layer.n_in, layer.n_out) for layer in cfg.model.layers.values()]
    assert layers == [(256, 128), (128, 12)]


def test_empty_value(initialize_hydra):
    cfg = compose(config_name="test_config_1.yml")
    assert not cfg.data.testing_data_path
