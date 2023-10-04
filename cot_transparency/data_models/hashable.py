from cot_transparency.util import deterministic_hash


from pydantic import BaseModel


class HashableBaseModel(BaseModel):
    def d_hash(self) -> str:
        as_json = self.model_dump_json()
        return deterministic_hash(as_json)
