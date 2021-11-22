from torch.utils.data import DataLoader

from data_generators import DataGenerator, DataGeneratorTRECIS


def test_initialize_data_generator(hydra_config):
    cfg = hydra_config
    generator = DataGeneratorTRECIS(cfg)
    assert isinstance(generator, DataGenerator)
    data_loader_train = generator(mode="train")
    data_loader_test = generator(mode="test")
    assert isinstance(data_loader_train, DataLoader)
    assert isinstance(data_loader_test, DataLoader)
    for batch in data_loader_train:
        assert len(batch["input_ids"]) == 1
