import logging
import warnings

from engines.trainer import MetaLearningTrainer, BatchLearningTrainer
from parsers import parse
from utils import instantiate_config


def get_trainer(config_name: str):
    if "few_shot" in config_name:
        return MetaLearningTrainer
    else:
        return BatchLearningTrainer


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    warnings.filterwarnings("ignore")

    args = parse()
    cfg = instantiate_config(args.config, args.job_name)
    trainer_class = get_trainer(args.config)
    trainer = trainer_class(cfg)
    trainer.run()


