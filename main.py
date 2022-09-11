import glob
import logging
import os
import warnings
from pathlib import Path
from time import sleep

import torch
from omegaconf import DictConfig, ListConfig

from event_extractor.engines.trainer import MetaLearningTrainer, BatchLearningTrainer
from event_extractor.parsers.parser import parse
from utils import instantiate_config


def get_trainer(config_name: str):
    if "few_shot" in config_name:
        return MetaLearningTrainer
    else:
        return BatchLearningTrainer


def run(cfg: DictConfig):
    if isinstance(cfg.seed, int):
        trainer = trainer_class(cfg)
        trainer.run()
        torch.cuda.empty_cache()
        sleep(10)
    elif isinstance(cfg.seed, ListConfig):
        for seed in cfg.seed:
            cfg.seed = seed
            trainer = trainer_class(cfg)
            trainer.run()
            torch.cuda.empty_cache()
            sleep(10)
    else:
        raise ValueError(f"Seed must be of type int or list of int.")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    warnings.filterwarnings("ignore")

    args = parse()
    if Path(args.config).is_file():
        cfg = instantiate_config(args.config)
        trainer_class = get_trainer(args.config)
        run(cfg)
    elif Path(args.config).is_dir():
        configs = glob.glob(str(Path(args.config)) + "/*.yaml")
        if not configs:
            configs = [os.path.join(path, name) for path, subdirs, files in os.walk(str(Path(args.config))) for name in files if
                     name.endswith(".yaml")]
        logger.warning(f"List of experiments to be run with the following configs: \n"
                       f"{configs}")
        for config in configs:
            cfg = instantiate_config(config)
            trainer_class = get_trainer(config)
            run(cfg)
    else:
        raise FileNotFoundError(f"{args.config} does not exist.")

