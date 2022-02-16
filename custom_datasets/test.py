from typing import Dict
import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split


class DataProcessorEventType(object):
    DATA_TO = {"train": "../../../data/trec_is/data/train.tsv",
               "test": "../../../data/trec_is/data/test.tsv"}
    INFO_FILE = {"train": "../../../data/trec_is/train.csv",
                 "test": "../../../data/trec_is/test.csv"}
    FILE_DIR = "../../../data/trec_is/data/json/"

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

    def create_df(self):
        pass

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
        df_train.to_csv(self.DATA_TO["train"], sep="\t", index=True, header=True)
        df_test.to_csv(self.DATA_TO["test"], sep="\t", index=True, header=True)
        return df_train, df_test


processor = DataProcessorEventType()
df_train, df_test = processor.run()
