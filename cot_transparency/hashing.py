import hashlib


def deterministic_hash(something: str) -> str:
    return hashlib.sha1(something.encode()).hexdigest()
