import logging
import warnings
import torch
from models.SingleLabelSequenceClassification import SingleLabelSequenceClassification
from data_generators import DataGenerator
from hydra import initialize, compose

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with initialize(config_path="./configs", job_name="test"):
        cfg = compose(config_name="example_config.yaml")
    generator = DataGenerator(cfg)
    data_loader_train = generator("train")
    data_loader_test = generator("test", batch_size=1)
    model = SingleLabelSequenceClassification(cfg, num_classes=generator.num_labels)
    model.train_model(data_loader_train)
    model = torch.load("./outputs/test_model.pt")
    model.test_model(data_loader_test)

