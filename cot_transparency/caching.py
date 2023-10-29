from pydantic import BaseModel
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel, write_jsonl_file_from_basemodel


from pathlib import Path
from threading import Lock
from typing import Callable, Generic, Self, Type, TypeVar


A = TypeVar("A", bound=BaseModel)


class FileCacheRow(Generic[A], BaseModel):
    key: str
    value: A


class BaseModelCacheWriter(Generic[A]):
    def __init__(
        self, cache_file: str | Path, base_model: Type[A], hash_func: Callable[[A], str], write_every: int = 50
    ):
        self.cache_file = Path(cache_file)
        self.lock = Lock()
        self.update_counter = 0
        self.cache: dict[str, A] = self._load_cache_dict()
        self.row_obj = FileCacheRow
        self.write_every = write_every
        self.hash_func = hash_func
        self.base_model = base_model

    def load(self) -> Self:
        self.cache = self._load_cache_dict()
        return self

    def write(self, x: A) -> A:
        """
        Write the object to the cache file if it is not already in the cache
        """
        key = self.hash_func(x)
        if key not in self.cache:
            self.cache[key] = x

        if self.update_counter % self.write_every == 0:
            self._write_cache_dict()
        self.update_counter += 1
        return x

    def _load_cache_dict(self):
        with self.lock:
            if self.cache_file.exists():
                cache_as_rows = read_jsonl_file_into_basemodel(self.cache_file, self.base_model)
                print(f"Loaded {len(cache_as_rows)} rows from cache file {self.cache_file.as_posix()}")
                cache = {row.key: row.value for row in cache_as_rows}
        return cache

    def _write_cache_dict(self):
        with self.lock:
            rows = [FileCacheRow[A](key=key, value=value) for key, value in self.cache.items()]
            write_jsonl_file_from_basemodel(self.cache_file, rows)

    def __del__(self):
        self._write_cache_dict()
