from custom_datasets import DatasetTRECIS
import pandas as pd


def test_dataset_creation(hydra_config):
    cfg = hydra_config
    sequences = ["Hallo ich bin zuhause."]
    df = pd.DataFrame({"sentence": sequences, "label": ["gold"]})
    dataset = DatasetTRECIS(df, cfg.data.sentence_column, cfg.data.label_column, cfg.model.from_pretrained)
    assert "input_ids" in dataset.encodings
    assert "token_type_ids" in dataset.encodings
    assert "attention_mask" in dataset.encodings
    assert len(dataset.encodings["input_ids"]) == 1 == len(dataset.encodings["token_type_ids"]) == len(dataset.encodings["attention_mask"])
    assert len(dataset.labels) == len(dataset.encodings["input_ids"])
