from pymongo import MongoClient
from dataclasses import dataclass,asdict
import numpy as np

@dataclass
class FaceVectorModel:
    vector: list
    label:str
    count:int


class DatabaseConnect:

    def __init__(self):
        client = MongoClient("mongodb://skyrim:***REDACTED***@47.103.29.129:27017")
        db = client["skyrim"]
        self. collection = db["face_vector"]

    def insert_one(self, data:FaceVectorModel):
        record = self.find_one({"label":data.label})
        if record:
            count = record["count"]+data.count
            vector = (np.array(record["vector"]) * record["count"] + np.array(data.vector) * data.count)/(record["count"]+data.count)
            vector = vector.tolist()
            self.collection.update_one({"label":data.label},{"$set":{"vector":vector,"count":count}})
            return
        else:
            self.collection.insert_one(asdict(data))

    def find_one(self,query:dict):
        record = self.collection.find_one(query)
        return record

    def find_all(self):
        records = self.collection.find()
        return records

