from custom_datasets import DatasetTRECIS
from data_generators import DataGenerator


class DataGeneratorTRECIS(DataGenerator):
    @property
    def training_dataset(self):
        return DatasetTRECIS(self.training_dataframe,
                             sentence_column=self.cfg.data.sentence_column,
                             label_column=self.cfg.data.label_column,
                             from_pretrained=self.cfg.model.from_pretrained)

    @property
    def testing_dataset(self):
        return DatasetTRECIS(self.testing_dataframe,
                             sentence_column=self.cfg.data.sentence_column,
                             label_column=self.cfg.data.label_column,
                             from_pretrained=self.cfg.model.from_pretrained)
