import os
from pathlib import Path
import tempfile
from typing import Optional, Sequence, Type, TypeVar

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


def read_jsonl_file_into_basemodel(path: Path | str, basemodel: Type[GenericBaseModel]) -> Slist[GenericBaseModel]:
    with open(path) as f:
        return Slist(
            caught_base_model_parse(basemodel=basemodel, line=line)
            for line in f.readlines()
            # filter for users
        )


def read_jsonl_file_into_basemodel_ignore_errors(
    path: Path, basemodel: Type[GenericBaseModel]
) -> Slist[GenericBaseModel]:
    with open(path) as f:
        return Slist(
            ignore_errors_base_model_parse(basemodel=basemodel, line=line)
            for line in f.readlines()
            # filter for users
        ).flatten_option()


class AtomicFile:
    def __init__(self, filename: str | Path):
        self.filename = Path(filename)
        self.dir_name = self.filename.parent

    def __enter__(self):
        self.temp_file = tempfile.NamedTemporaryFile("w", dir=self.dir_name, delete=False)
        return self.temp_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the file if it's open
        self.temp_file.close()

        try:
            # if all went well we can rename the temp file
            if exc_type is None:
                os.replace(self.temp_file.name, self.filename)
            else:
                os.remove(self.temp_file.name)
        except Exception as e:
            raise e
        finally:
            # Cleanup, in case the removal did not happen
            if os.path.exists(self.temp_file.name):
                os.remove(self.temp_file.name)


def write_jsonl_file_from_basemodel(path: Path | str, basemodels: Sequence[BaseModel]) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with AtomicFile(path) as f:
        for basemodel in basemodels:
            f.write(basemodel.model_dump_json() + "\n")


def write_csv_file_from_basemodel(path: Path, basemodels: Sequence[BaseModel]) -> None:
    """Uses pandas"""
    df = pd.DataFrame([model.model_dump() for model in basemodels])
    df.to_csv(path)


def read_base_model_from_csv(path: Path, basemodel: Type[GenericBaseModel]) -> Slist[GenericBaseModel]:
    df = pd.read_csv(path)
    return Slist(basemodel(**row) for _, row in df.iterrows())


def safe_file_write(filename: str | Path, data: str):
    with AtomicFile(filename) as f:
        f.write(data)
