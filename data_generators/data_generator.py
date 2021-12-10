from typing import Optional, Dict

from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Sampler, RandomSampler


class DataGenerator(object):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.num_labels = self.training_dataset.features['label'].num_classes
        self.label_index_map: Dict = {label: self.training_dataset.features['label'].str2int(label)
                                      for label in self.training_dataset.features['label'].names}

    @property
    def training_dataset(self):
        return load_dataset(self.cfg.data.name, split='train')

    @property
    def testing_dataset(self):
        return load_dataset(self.cfg.data.name, split='test')

    def __call__(self,
                 mode: str,
                 batch_size: Optional[int] = None,
                 sampler: Optional[Sampler] = None) -> DataLoader:
        """
        :param mode: train, valid or test
        :param batch_size:
        :param shuffle:
        :param sampler:
        :return:
        """
        batch_size = batch_size or self.cfg.data.batch_size
        if mode == "train":
            dataset = self.training_dataset
        elif mode == "test":
            dataset = self.testing_dataset
        else:
            raise AttributeError(f"{mode} is not a valid attribute in Data Generator class.")
        # default sampler is a random sampler over all entries in the dataset.
        sampler = sampler or RandomSampler(dataset)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler) #TODO: add sampler option


class DataGeneratorSubSample(DataGenerator):
    @property
    def training_dataset(self):
        return load_dataset(self.cfg.data.name, split='train').train_test_split(test_size=0.6)["train"]

    @property
    def testing_dataset(self):
        return load_dataset(self.cfg.data.name, split='test').train_test_split(test_size=0.4)["test"]