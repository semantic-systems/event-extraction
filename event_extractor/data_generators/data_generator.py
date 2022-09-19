from typing import Optional, Dict

from datasets import load_dataset, concatenate_datasets, ClassLabel
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
        dataset = load_dataset(self.cfg.data.name, self.cfg.data.config, split='train')
        if self.cfg.data.label_column != 'label':
            dataset = self.rename_label_column(dataset, self.cfg.data.label_column, 'label')
        if "TRECIS_event_type" in self.cfg.data.name:
            dataset_with_oos = concatenate_datasets([dataset, self.oos_dataset("train")])
            new_features = dataset_with_oos.features.copy()
            new_features["label"] = ClassLabel(names=dataset_with_oos.features["label"].names + ["oos"],
                                               num_classes=dataset_with_oos.features["label"].num_classes + 1)
            dataset = dataset_with_oos.cast(new_features)
        return dataset

    @property
    def validation_dataset(self):
        dataset = load_dataset(self.cfg.data.name, self.cfg.data.config, split='validation')
        if self.cfg.data.label_column != 'label':
            dataset = self.rename_label_column(dataset, self.cfg.data.label_column, 'label')
        if "TRECIS_event_type" in self.cfg.data.name:
            dataset_with_oos = concatenate_datasets([dataset, self.oos_dataset("validation")])
            new_features = dataset_with_oos.features.copy()
            new_features["label"] = ClassLabel(names=dataset_with_oos.features["label"].names + ["oos"],
                                               num_classes=dataset_with_oos.features["label"].num_classes + 1)
            dataset = dataset_with_oos.cast(new_features)
        return dataset

    @property
    def testing_dataset(self):
        dataset = load_dataset(self.cfg.data.name, self.cfg.data.config, split='test')
        if self.cfg.data.label_column != 'label':
            dataset = self.rename_label_column(dataset, self.cfg.data.label_column, 'label')
        if "TRECIS_event_type" in self.cfg.data.name:
            dataset_with_oos = concatenate_datasets([dataset, self.oos_dataset("test")])
            new_features = dataset_with_oos.features.copy()
            new_features["label"] = ClassLabel(names=dataset_with_oos.features["label"].names + ["oos"],
                                               num_classes=dataset_with_oos.features["label"].num_classes + 1)
            dataset = dataset_with_oos.cast(new_features)
        return dataset

    @staticmethod
    def oos_dataset(mode: str):
        def change_label(example):
            example["label"] = 9
            return example
        datasets = [load_dataset("tweet_eval", "emoji", split=mode),
                    load_dataset("tweet_eval", "emotion", split=mode),
                    load_dataset("tweet_eval", "sentiment", split=mode),
                    load_dataset("tweet_eval", "hate", split=mode),
                    load_dataset("tweet_eval", "offensive", split=mode),
                    load_dataset("tweet_eval", "irony", split=mode),
                    load_dataset("tweet_eval", "stance_abortion", split=mode),
                    load_dataset("tweet_eval", "stance_atheism", split=mode),
                    load_dataset("tweet_eval", "stance_climate", split=mode),
                    load_dataset("tweet_eval", "stance_feminist", split=mode),
                    load_dataset("tweet_eval", "stance_hillary", split=mode)
                    ]
        concatenated_dataset = concatenate_datasets(datasets)
        updated_dataset = concatenated_dataset.map(change_label)
        new_features = updated_dataset.features.copy()
        new_features["label"] = ClassLabel(names=["oos"], num_classes=1)
        updated_dataset = updated_dataset.cast(new_features)
        return updated_dataset

    @staticmethod
    def rename_label_column(dataset, original_label_name, new_label_name):
        return dataset.rename_column(original_label_name, new_label_name)

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
        elif mode == "validation":
            dataset = self.validation_dataset
        else:
            raise AttributeError(f"{mode} is not a valid attribute in Data Generator class.")
        # default sampler is a random sampler over all entries in the dataset.
        sampler = sampler or RandomSampler(dataset)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


class DataGeneratorSubSample(DataGenerator):
    @property
    def training_dataset(self):
        dataset = load_dataset(self.cfg.data.name, self.cfg.data.config, split='train').train_test_split(test_size=self.cfg.data.subset)["test"]
        if self.cfg.data.label_column != 'label':
            dataset = self.rename_label_column(dataset, self.cfg.data.label_column, 'label')
        return dataset

    @property
    def testing_dataset(self):
        dataset = load_dataset(self.cfg.data.name, self.cfg.data.config, split='test').train_test_split(test_size=self.cfg.data.subset)["test"]
        if self.cfg.data.label_column != 'label':
            dataset = self.rename_label_column(dataset, self.cfg.data.label_column, 'label')
        return dataset

    @property
    def validation_dataset(self):
        dataset = load_dataset(self.cfg.data.name, self.cfg.data.config, split='validation').train_test_split(test_size=self.cfg.data.subset)["test"]
        if self.cfg.data.label_column != 'label':
            dataset = self.rename_label_column(dataset, self.cfg.data.label_column, 'label')
        return dataset
