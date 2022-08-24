import os
from typing import Dict, Any, List

import yaml
import glob


class ConfigWriter(object):
    @staticmethod
    def write_from_dict(dict_to_dump: Dict, path_to_dump: str):
        with open(path_to_dump, 'w') as f:
            yaml.dump(dict_to_dump, f, default_flow_style=False)

    @staticmethod
    def read_yaml(yaml_to_read: str) -> Dict:
        with open(yaml_to_read, 'r') as f:
            data = yaml.safe_load(f)
        return data

    @staticmethod
    def change_field_of_all(dir: str):
        files = [os.path.join(path, name) for path, subdirs, files in os.walk(dir) for name in files if name.endswith(".yaml")]
        updated_dicts: List[Dict] = []
        for file in files:
            config = ConfigWriter.read_yaml(file)
            config["seed"] = 1
            updated_dicts.append(config)
            ConfigWriter.write_from_dict(config, file)
        # print(updated_dicts)


if __name__ == "__main__":
    ConfigWriter.change_field_of_all("./event_extractor/configs/tweeteval/")
