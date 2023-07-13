import datetime
import os
from typing import Optional


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
