# from datasets import load_dataset
from models.SingleLabelSequenceClassification import SingleLabelSequenceClassification
from data_generators import DataGeneratorTRECIS
from hydra import initialize, compose


if __name__ == "__main__":
    with initialize(config_path="./configs", job_name="test"):
        cfg = compose(config_name="example_config.yaml")
    # dataset = load_dataset("tweet_eval", "emoji", split='train')
    generator = DataGeneratorTRECIS(cfg)
    data_loader_train = generator("train")
    model = SingleLabelSequenceClassification(cfg.model)
    for batch in enumerate(data_loader_train):
        model(batch["sentence"], batch["attention_mask"], None)
