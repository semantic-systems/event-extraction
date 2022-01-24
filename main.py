import logging
import warnings

from engines.trainer import MetaLearningTrainer
from parsers import parse
from utils import instantiate_config


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    warnings.filterwarnings("ignore")

    args = parse()
    cfg = instantiate_config(args.config, args.job_name)
    trainer = MetaLearningTrainer(cfg)
    trainer.run()


