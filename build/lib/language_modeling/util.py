import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import pkg_resources
from box import Box
import yaml
from language_modeling.Project import Project
import torch


logger = logging.getLogger("language_modeling")


def get_resource_string(path: str, decode=True) -> Union[str, bytes]:
    """
    Load a package resource (i.e. a file from within this package)

    :param path: the path, starting at the root of the current module (e.g. 'res/default.conf').
           must be a string, not a Path object!
    :param decode: if true, decode the file contents as string (otherwise return bytes)
    :return: the contents of the resource file (as string or bytes)
    """
    s = pkg_resources.resource_string(__name__.split(".")[0], path)
    return s.decode(errors="ignore") if decode else s


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load the config from the specified yaml file

    :param config_file: path of the config file to load
    :return: the parsed config as dictionary
    """
    with open(config_file, "r") as fp:
        return yaml.safe_load(fp)


def logging_setup(config: Dict):
    """
    setup logging based on the configuration

    :param config: the parsed config tree
    """
    log_conf = config["logging"]
    fmt = log_conf["format"]
    if log_conf["enabled"]:
        level = logging._nameToLevel[log_conf["level"].upper()]
    else:
        level = logging.NOTSET
    logging.basicConfig(format=fmt, level=logging.WARNING)
    logger.setLevel(level)


def get_default_logger() -> logging.Logger:
    config_file = str(Project.main_config_file)
    config = load_config(config_file)

    logging_setup(config)
    return logger


def load_hparams() -> Box:
    return Box.from_yaml(filename=Project().hparams_file)


def print_top_n_tokens(text: str, model, tokenizer, n=5) -> Dict:
    print(text)
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(**inputs).logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, n, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")


if __name__ == "__main__":
    print(load_hparams())
