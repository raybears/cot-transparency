import datetime
import os
from typing import Optional
import logging


def get_exp_dir_name(
    exp_dir: Optional[str] = None,
    experiment_suffix: Optional[str] = None,
    sub_dir: Optional[str] = None,
    root: str = "./experiments",
):
    if exp_dir is None:
        root = "./experiments"
        if sub_dir:
            root = os.path.join(root, sub_dir)
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_dir = os.path.join(root, now)
    if experiment_suffix:
        exp_dir += f"_{experiment_suffix}"
    return exp_dir


def setup_logger(name: str, level: int = logging.WARNING):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
