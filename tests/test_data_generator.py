from torch.utils.data import DataLoader

from data_generators import DataGenerator, DataGeneratorTRECIS


def test_initialize_data_generator(hydra_config):
    cfg = hydra_config
    generator = DataGeneratorTRECIS(cfg)
    assert isinstance(generator, DataGenerator)
    assert isinstance(generator(mode="train", batch_size=cfg.data.batch_size), DataLoader)
    assert isinstance(generator(mode="test", batch_size=cfg.data.batch_size), DataLoader)
