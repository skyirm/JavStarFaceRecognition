from dataclasses import dataclass


@dataclass
class FaceVectorModel:
    vector: list
    label:str
    count:int
    thumb: bytes | None = None