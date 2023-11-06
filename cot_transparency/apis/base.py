from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Self, Sequence

from pydantic import BaseModel

from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.json_utils.read_write import (
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from cot_transparency.util import deterministic_hash


class ModelType(str, Enum):
    # moves the "assistant preferred" message to the previous (user) message
    chat = "chat"
    completion = "completion"
    chat_with_append_assistant = "anthropic"

    @staticmethod
    def from_model_name(name: str) -> "ModelType":
        if "claude" in name:
            return ModelType.chat_with_append_assistant
        elif "gpt-3.5-turbo" in name or name == "gpt-4" or name == "gpt-4-32k":
            return ModelType.chat
        else:
            return ModelType.completion


class Prompt(BaseModel):
    """
    You shouldn't really need to use this class directly, unless you are debugging or testing
    the specific format that is being sent to an api call. ModelCallers take a
    list of ChatMessages but use subtypes of this class to format the messages into the
    format that the api expects
    """

    def __str__(self) -> str:
        out = ""
        for msg in self.messages:
            out += f"\n\n{msg.role}\n{msg.content}"
        return out

    messages: Sequence[ChatMessage]

    @classmethod
    def from_prompt(cls, prompt: "Prompt") -> Self:
        return cls(messages=prompt.messages)

    def __add__(self, other: Self) -> Self:
        return Prompt(messages=list(self.messages) + list(other.messages))


class InferenceResponse(BaseModel):
    raw_responses: Sequence[str]

    @property
    def has_multiple_responses(self) -> bool:
        return len(self.raw_responses) > 1

    @property
    def single_response(self) -> str:
        if self.has_multiple_responses:
            raise ValueError("This response has multiple responses")
        else:
            return self.raw_responses[0]


class ModelCaller(ABC):
    @abstractmethod
    def call(
        self,
        messages: Sequence[ChatMessage],
        config: OpenaiInferenceConfig,
    ) -> InferenceResponse:
        raise NotImplementedError()

    def with_file_cache(self, cache_path: Path | str, write_every_n: int = 20) -> "CachedCaller":
        """
        Load a file cache from a path
        Alternatively, rather than write_every_n, just dump with append mode?
        """
        if isinstance(cache_path, str):
            cache_path = Path(cache_path)

        return CachedCaller(wrapped_caller=self, cache_path=cache_path, write_every_n=write_every_n)

    def with_model_specific_file_cache(self, cache_dir: Path | str, write_every_n: int = 20) -> "CachedPerModelCaller":
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        assert not (cache_dir.is_file() or cache_dir.suffix == ".jsonl"), "Cache dir must be a directory"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return CachedPerModelCaller(wrapped_caller=self, cache_dir=cache_dir, write_every_n=write_every_n)


class CachedValue(BaseModel):
    response: InferenceResponse
    messages: Sequence[ChatMessage]
    config: OpenaiInferenceConfig


class FileCacheRow(BaseModel):
    key: str
    response: CachedValue


class APIRequestCache:
    def __init__(self, cache_path: Path | str):
        self.cache_path = Path(cache_path)
        self.data: dict[str, CachedValue] = {}
        self.load()

    def load(self, silent: bool = False) -> Self:
        """
        Load a file cache from a path
        """

        if self.cache_path.exists():
            rows = read_jsonl_file_into_basemodel(
                path=self.cache_path,
                basemodel=FileCacheRow,
            )
            if not silent:
                print(f"Loaded {len(rows)} rows from cache file {self.cache_path.as_posix()}")
            self.data = {row.key: row.response for row in rows}
        return self

    def save(self) -> None:
        """
        Save a file cache to a path
        """
        rows = [FileCacheRow(key=key, response=response) for key, response in self.data.items()]
        write_jsonl_file_from_basemodel(self.cache_path, rows)

    def __getitem__(self, key: str) -> CachedValue:
        return self.data[key]

    def __setitem__(self, key: str, value: CachedValue) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data


def file_cache_key(messages: Sequence[ChatMessage], config: OpenaiInferenceConfig) -> str:
    str_messages = ",".join([str(msg) for msg in messages]) + config.model_hash()
    return deterministic_hash(str_messages)


class CachedCaller(ModelCaller):
    def __init__(self, wrapped_caller: ModelCaller, cache_path: Path, write_every_n: int, silent_loading: bool = False):
        self.model_caller = wrapped_caller
        self.cache_path = cache_path
        self.cache: APIRequestCache = APIRequestCache(cache_path)
        self.write_every_n = write_every_n
        self.__update_counter = 0
        self.save_lock = Lock()

    def save_cache(self) -> None:
        self.cache.save()

    def call(
        self,
        messages: Sequence[ChatMessage],
        config: OpenaiInferenceConfig,
    ) -> InferenceResponse:
        key = file_cache_key(messages, config)
        if key in self.cache:
            return self.cache[key].response
        else:
            response = self.model_caller.call(messages, config)
            value = CachedValue(
                response=response,
                messages=messages,
                config=config,
            )
            with self.save_lock:
                self.cache[key] = value
                self.__update_counter += 1
                if self.__update_counter % self.write_every_n == 0:
                    self.save_cache()
            return response

    def __del__(self):
        self.save_cache()


class CachedPerModelCaller(ModelCaller):
    """
    This class will create a seperate cache (and corresponding file) for each model that is called
    useful if you want to run multiple models in parallel without having to worry about cache conflicts
    """

    def __init__(self, wrapped_caller: ModelCaller, cache_dir: Path, write_every_n: int, preload: bool = False):
        self.model_caller = wrapped_caller
        self.cache_dir = cache_dir
        self.cache_callers: dict[str, CachedCaller] = {}
        self.write_every_n = write_every_n
        self.lock = Lock()
        if preload:
            cache_paths = self.cache_dir.glob("*.jsonl")
            for path in cache_paths:
                cache_name = path.stem
                self._create_caller(cache_path=path, cache_name=cache_name)

    def _create_caller(self, cache_path: Path, cache_name: str) -> CachedCaller:
        self.cache_callers[cache_name] = CachedCaller(
            wrapped_caller=self.model_caller,
            cache_path=cache_path,
            write_every_n=self.write_every_n,
            silent_loading=True,
        )
        return self.cache_callers[cache_name]

    def call(
        self,
        messages: Sequence[ChatMessage],
        config: OpenaiInferenceConfig,
    ) -> InferenceResponse:
        return self.get_specific_caller(config.model).call(messages, config)

    def get_specific_caller(
        self,
        cache_name: str,
    ) -> CachedCaller:
        """
        This returns a CachedCaller that will save into self.cache_dir/cache_name.jsonl.
        A reference to this caller is stored in self.cache_callers
        """

        with self.lock:
            if cache_name not in self.cache_callers:
                cache_path = self.cache_dir / f"{cache_name}.jsonl"
                self._create_caller(cache_path=cache_path, cache_name=cache_name)

        return self.cache_callers[cache_name]
