from torch.utils.data import DataLoader
from custom_datasets import load_dataset
from models.SingleLabelSequenceClassification import SingleLabelSequenceClassification
from hydra import initialize, compose


if __name__ == "__main__":
    initialize(config_path="./configs", job_name="test")
    cfg = compose(config_name="example_config.yml")

    dataset = load_dataset("tweet_eval", "emoji", split='train')
    data_loader = DataLoader(dataset, shuffle=True, batch_size=8)
    model = SingleLabelSequenceClassification(cfg.model)
    for i, (sentence, label) in enumerate(data_loader):
        model(sentence)