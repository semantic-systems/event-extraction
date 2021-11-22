from custom_datasets import CustomDataset
from transformers import PreTrainedTokenizer, BertTokenizer
from torch import tensor


class DatasetTRECIS(CustomDataset):
    # @property
    # def tokenizer(self):
    #
    #     return tokenizer

    def preprocess(self) -> tensor:
        tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained(self.from_pretrained)
        # TODO: add padding and truncation and max_length parameters.
        tokenized_sentences: tensor = tokenizer(self.df[self.sentence_column].tolist(),
                                                     padding=True,
                                                     truncation=True,
                                                     return_tensors="pt")
        return tokenized_sentences
