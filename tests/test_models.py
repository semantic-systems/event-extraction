

def test_initialize_model(model_instance):
    assert model_instance
    assert len(model_instance.bert.encoder.layer) == 2
