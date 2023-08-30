from pathlib import Path
from typing import Type, TypeVar, Sequence, Optional

import pandas as pd
from pydantic import BaseModel
from slist import Slist

GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


def caught_base_model_parse(basemodel: Type[GenericBaseModel], line: str) -> GenericBaseModel:
    try:
        return basemodel.model_validate_json(line)
    except Exception as e:
        print(f"Error parsing line: {line}")
        raise e


def ignore_errors_base_model_parse(basemodel: Type[GenericBaseModel], line: str) -> Optional[GenericBaseModel]:
    try:
        return basemodel.parse_raw(line)
    except Exception:
        return None


def read_jsonl_file_into_basemodel(path: Path, basemodel: Type[GenericBaseModel]) -> Slist[GenericBaseModel]:
    with open(path, "r") as f:
        return Slist(
            caught_base_model_parse(basemodel=basemodel, line=line)
            for line in f.readlines()
            # filter for users
        )


def read_jsonl_file_into_basemodel_ignore_errors(
    path: Path, basemodel: Type[GenericBaseModel]
) -> Slist[GenericBaseModel]:
    with open(path, "r") as f:
        return Slist(
            ignore_errors_base_model_parse(basemodel=basemodel, line=line)
            for line in f.readlines()
            # filter for users
        ).flatten_option()


def write_jsonl_file_from_basemodel(path: Path, basemodels: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for basemodel in basemodels:
            f.write(basemodel.json() + "\n")


def write_csv_file_from_basemodel(path: Path, basemodels: Sequence[BaseModel]) -> None:
    """Uses pandas"""
    df = pd.DataFrame([model.model_dump() for model in basemodels])
    df.to_csv(path)


def read_base_model_from_csv(path: Path, basemodel: Type[GenericBaseModel]) -> Slist[GenericBaseModel]:
    df = pd.read_csv(path)
    return Slist(basemodel(**row) for _, row in df.iterrows())
