"""_summary_
"""
import yaml
from loguru import logger


class YMLLoader:
    """_summary_
    """
    def __init__(self, *args, **kwargs) -> None:
        self.path = kwargs.get("path") or '.'
        self.args = args  # Remove this

        try:
            file = open(self.path, "r")
        except FileNotFoundError as err:
            logger.exception(
                f"The file at path '{self.path}' is not found!! --> {err}")
        else:
            self.configs = yaml.safe_load(file)

        # TODO:
        # 1. Handle other exceptions


class Configs(YMLLoader):
    """_summary_

    Args:
        YMLLoader (_type_): _description_
    """
    def __init__(self, *args, **kwargs) -> None:
        super(Configs, self).__init__(*args, **kwargs)
        pass
    def dump(self, ):
        try:
            with open(self.path, "w", encoding="utf-8") as file:
                file.write(yaml.dump(self.configs))
        except FileNotFoundError as err:
            logger.exception(f"Exception while updating the configs : {err}")

if __name__ == "__main__":
    configs = Configs(path="../configs.yml")
    logger.info(configs.configs)
    