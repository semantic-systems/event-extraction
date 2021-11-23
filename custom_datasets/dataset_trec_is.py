from custom_datasets import CustomDataset
from transformers import PreTrainedTokenizer, BertTokenizer
from torch import tensor
from sklearn import preprocessing


class DatasetTRECIS(CustomDataset):

    def preprocess(self) -> tensor:
        tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained(self.from_pretrained)
        # TODO: add padding and truncation and max_length parameters.
        tokenized_sentences: tensor = tokenizer(self.df[self.sentence_column].tolist(),
                                                padding=True,
                                                truncation=True,
                                                return_tensors="pt")
        le = preprocessing.LabelEncoder()
        self.labels = tensor(le.fit_transform(self.df[self.label_column].tolist()))
        index = le.fit_transform(self.df[self.label_column].unique())
        label = le.inverse_transform(index)
        self.label_index_map = dict(zip(label, index))

        return tokenized_sentences
