from pathlib import Path
from typing import Type, TypeVar, Sequence

from pydantic import BaseModel
from slist import Slist

GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


def caught_base_model_parse(basemodel: Type[GenericBaseModel], line: str) -> GenericBaseModel:
    try:
        return basemodel.parse_raw(line)
    except Exception as e:
        print(f"Error parsing line: {line}")
        raise e


def read_jsonl_file_into_basemodel(path: Path, basemodel: Type[GenericBaseModel]) -> Slist[GenericBaseModel]:
    with open(path, "r") as f:
        return Slist(
            caught_base_model_parse(basemodel=basemodel, line=line)
            for line in f.readlines()
            # filter for users
        )


def write_jsonl_file_from_basemodel(path: Path, basemodels: Sequence[BaseModel]) -> None:
    with open(path, "w") as f:
        for basemodel in basemodels:
            f.write(basemodel.json() + "\n")
