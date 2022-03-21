from torch.utils.data import DataLoader

from event_extractor.data_generators import DataGenerator


def test_initialize_data_generator(hydra_config):
    cfg = hydra_config
    generator = DataGenerator(cfg)
    assert isinstance(generator, DataGenerator)
    data_loader_train = generator(mode="train")
    data_loader_test = generator(mode="test")
    assert isinstance(data_loader_train, DataLoader)
    assert isinstance(data_loader_test, DataLoader)
    for batch in data_loader_train:
        assert len(batch["text"]) == 1
