from hydra import compose
from models.SingleLabelSequenceClassification import SingleLabelSequenceClassification


def test_initialize_model():
    cfg = compose(config_name="test_config_1.yml")
    model = SingleLabelSequenceClassification(cfg.model)
    assert model
    assert len(model.bert) == 1
