import glob
import logging
import warnings
from pathlib import Path
from event_extractor.engines.trainer import MetaLearningTrainer, BatchLearningTrainer
from event_extractor.parsers.parser import parse
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
    if Path(args.config).is_file():
        cfg = instantiate_config(args.config, args.job_name)
        trainer_class = get_trainer(args.config)
        trainer = trainer_class(cfg)
        trainer.run()
    elif Path(args.config).is_dir():
        configs = glob.glob(str(Path(args.config).absolute()) + "*.yaml")
        for config in configs:
            cfg = instantiate_config(config, args.job_name)
            trainer_class = get_trainer(config)
            trainer = trainer_class(cfg)
            trainer.run()
    else:
        pass

