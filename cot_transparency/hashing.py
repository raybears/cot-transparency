import hashlib

from pydantic import BaseModel


def deterministic_hash(something: str) -> str:
    return hashlib.sha1(something.encode()).hexdigest()


class HashableBaseModel(BaseModel):
    def d_hash(self) -> str:
        as_json = self.json()
        return deterministic_hash(as_json)
