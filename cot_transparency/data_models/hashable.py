from pydantic import BaseModel

from cot_transparency.util import deterministic_hash


class HashableBaseModel(BaseModel):
    def model_hash(self) -> str:
        as_json = self.model_dump_json()
        return deterministic_hash(as_json)

    class Config:
        # this is needed for the hashable base model
        frozen = True
