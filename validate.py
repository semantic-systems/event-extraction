# validate config value
import logging

from omegaconf import DictConfig


class ConfigValidator(object):
    def __init__(self, config: DictConfig):
        self.config = config
        self.model = config.model
        self.name = config.name
        self.data = config.data
        self.episode = config.episode

    def validate(self):
        try:
            self.validate_model()
            self.validate_data()
            self.validate_episode()
        except ValueError:
            logging.error(f"Validation of config failed.")
            raise

    def validate_model(self):
        pass

    def validate_data(self):
        pass

    def validate_episode(self):
        pass


