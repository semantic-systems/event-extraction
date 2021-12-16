import logging
from pathlib import Path

from omegaconf import DictConfig


class ConfigValidator(object):
    def __init__(self, config: DictConfig):
        self.config = config

    def __call__(self):
        try:
            self.validate_model()
            self.validate_data()
            self.validate_episode()
        except ValueError:
            logging.error(f"Validation of config failed.")
            raise

    def validate_model(self):
        self.create_output_path()

    def validate_data(self):
        pass

    def validate_episode(self):
        pass

    def create_output_path(self):
        if not Path(self.config.model.output_path, self.config.name).absolute().exists():
            logging.warning(f"Output path {str(Path(self.config.model.output_path, self.config.name).absolute())} "
                            f"does not exist. It will be automatically created. ")
            Path(self.config.model.output_path, self.config.name).absolute().mkdir(parents=True, exist_ok=True)
        if not Path(self.config.model.output_path, self.config.name, "pretrained_models").absolute().exists():
            Path(self.config.model.output_path, self.config.name, "pretrained_models").absolute().mkdir(parents=True, exist_ok=True)
        if not Path(self.config.model.output_path, self.config.name, "plots").absolute().exists():
            Path(self.config.model.output_path, self.config.name, "plots").absolute().mkdir(parents=True, exist_ok=True)



