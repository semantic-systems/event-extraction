import csv
import os
import glob
import datasets
import pandas as pd
import numpy as np

from typing import Dict
from datasets.tasks import TextClassification
from sklearn.model_selection import train_test_split


_CITATION = """\

"""

_DESCRIPTION = """\
 TREC Incident Stream Dataset: REC-IS provides multiple Twitter datasets collected from a range of past wildfire, 
 earthquake, flood, typhoon/hurricane, bombing and shooting events (labeled pandemic data will be added later this 
 year). We have had human annotators manually label this data into 25 information types based on the information each 
 tweet contains, such as 'contains location' or is a 'search and rescue request'. Each tweet is also assigned a 
 'priority' label, that indicates how critical the information within that tweet is for a response officer to see. 
 To date, TREC-IS has manually annotated tweet streams for 48 emergency events, comprising 50,000 tweets and
  producing over 125,000 labels.
"""


class TrecisConfig(datasets.BuilderConfig):
    """BuilderConfig for TREC IS."""

    def __init__(self, **kwargs):
        """BuilderConfig for TREC IS.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TrecisConfig, self).__init__(**kwargs)


class DataProcessorEventType(object):
    DATA_TO = {"train": "../../../../data/trec_is/data/train.csv",
               "test": "../../../../data/trec_is/data/test.csv"}
    INFO_FILE = {"train": "../../../../data/trec_is/train.csv",
                 "test": "../../../../data/trec_is/test.csv"}
    FILE_DIR = "../../../../../data/trec_is/data/json/"

    def __init__(self):
        self.data = {}
        self.label_dict = self.create_label_dict()

    def create_label_dict(self) -> Dict:
        train = pd.read_csv(self.INFO_FILE["train"], sep=';')
        test = pd.read_csv(self.INFO_FILE["test"], sep=';')
        train_file_list = train["num"].tolist()
        text_file_list = test["num"].tolist()
        train_label = train["label"].tolist()
        test_label = test["label"].tolist()
        label_dict = dict(zip(train_file_list+text_file_list, train_label+test_label))
        original_key = label_dict.keys()
        new_key = [f'{int(key):03d}' if "-" not in key else f'{int(key.split("-")[-1]):03d}' for key in original_key]
        label_dict = dict(zip(new_key, train_label+test_label))
        return label_dict

    def get_file_list(self) -> list:
        return glob.glob(self.FILE_DIR + "*.json")

    def run(self):
        data = {"text": [], "label": []}
        file_list = self.get_file_list()
        for file in file_list:
            file_dict = pd.read_json(file, lines=True)
            try:
                data["text"].extend(file_dict["text"])
                text_count = len(file_dict["text"])
            except KeyError:
                data["text"].extend(file_dict["full_text"])
                text_count = len(file_dict["full_text"])
            data["label"].extend([self.label_dict[file.split('-')[-1].split('.')[0]]]*text_count)
        x_train, x_test, y_train, y_test = train_test_split(
            np.asarray(data["text"]),
            np.asarray(data["label"]),
            test_size=0.33,
            random_state=4
        )
        df_train = pd.DataFrame({'text': x_train, 'label': y_train})
        df_test = pd.DataFrame({'text': x_test, 'label': y_test})
        df_train.to_csv(self.DATA_TO["train"], sep=";", index=True, header=True)
        df_test.to_csv(self.DATA_TO["test"], sep=";", index=True, header=True)
        return df_train, df_test


class TrecisDataset(datasets.GeneratorBasedBuilder):
    """TREC-IS Dataset"""

    _URL = {"train": "../../../../data/trec_is/train_final_clean.tsv",
            "validation": "../../../../data/trec_is/validation_final_clean.tsv",
            "test": "../../../../data/trec_is/test_final_clean.tsv"}

    train_data_label = ['tropical_storm', 'flood', 'shooting', 'covid', 'earthquake', 'hostage', 'fire', 'wildfire', 'explosion']
    BUILDER_CONFIGS = [
        TrecisConfig(
            name="trecis",
            version=datasets.Version("2.0.0"),
            description="TREC-IS Dataset: TREC Incident Stream Dataset",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['tropical_storm', 'flood', 'shooting', 'covid',
                                                                 'earthquake', 'hostage', 'fire', 'wildfire', 'explosion']),
                }
            ),
            supervised_keys=None,
            homepage="http://dcs.gla.ac.uk/~richardm/TREC_IS/2020/data.html",
            citation=_CITATION,
            task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(self._URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir["train"])}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_dir["validation"])}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir["test"])}
            )
        ]

    def _generate_examples(self, filepath):
        """Generate Trec IS examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter="\t", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            # remove header from the generator
            _ = next(csv_reader)
            for id_, row in enumerate(csv_reader):
                try:
                    _, text, label = row
                    yield id_, {"text": text, "label": label}
                except ValueError:
                    # TODO: log and check the invalid entries
                    pass


if __name__ == "__main__":
    processor = DataProcessorEventType()
    _, _ = processor.run()
