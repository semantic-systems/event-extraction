import torch


def test_initialize_model(model_instance):
    assert model_instance
    assert len(model_instance.bert.encoder.layer) == 2


def test_tokenize(model_instance):
    sequences = ["Hallo ich bin zuhause."]
    tokenized = model_instance.tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    assert "attention_mask" in tokenized
    assert "input_ids" in tokenized
    assert "token_type_ids" in tokenized


def test_model_forward(model_instance):
    sequences = ["Hallo ich bin zuhause."]
    tokens = model_instance.tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    embeddings = model_instance(tokens["input_ids"], tokens["attention_mask"])
    assert embeddings.shape == torch.Size([1, 20])