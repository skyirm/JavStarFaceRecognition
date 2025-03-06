import numpy as np
from peewee import Model, PostgresqlDatabase,TextField,IntegerField
from pgvector.peewee import VectorField

from logger import get_logger
from model import FaceVectorModel

logger = get_logger(__name__)

db = PostgresqlDatabase(
            'skyrim',
            user='skyrim',
            password='***REDACTED***',
            host='***REDACTED***',
            port=5432
        )

class Item(Model):
    name = TextField(unique=True)
    vector = VectorField(dimensions=512)
    count = IntegerField()

    class Meta:
        database = db
        table_name = "face_vector"

class PostgresConnection:
    def __init__(self):
        self.db = db
        try:
            self.db.connect()
        except:
            logger.critical("Cannot connect to database")
        self.Item = Item

    def insert_one(self, data:FaceVectorModel):
        record = self.find_one_by_name(data)
        if record:
            record.vector = data.vector
            record.count = data.count
            record.save()
            return
        else:
            self.Item.create(name=data.label,vector=data.vector,count=data.count)
        logger.info("Insert %s into database", data.label)

    def find_one_by_name(self, data:FaceVectorModel):
        record =  self.Item.get_or_none(self.Item.name == data.label)
        if record:
            return record
        else:
            return None

    def update_one(self, data:FaceVectorModel):
        record = self.find_one_by_name(data)
        if record:
            count = record.count+data.count
            vector = ((np.array(record.vector)*record.count + np.array(data.vector)*data.count)/(record.count+data.count)).tolist()
            record.vector = vector
            record.count = count
            record.save()
        else:
            self.Item.create(name=data.label,vector=data.vector,count=data.count)
        logger.info("Update %s in database", data.label)

    def find_most_similar(self,vector:list[float]):
        similar_items = (
            self.Item.select(
                self.Item.name,
                self.Item.vector.cosine_distance(vector).alias("similarity"),
                self.Item.count
            ).order_by( self.Item.vector.cosine_distance(vector)).limit(1)
        )
        return similar_items[0].name, similar_items[0].similarity

    def find_all_name(self):
        records =  self.Item.select( self.Item.name)
        return [record.name for record in records]

    def check_connection(self, func):
        def wrapper(*args, **kwargs):
            if self.db.is_closed():
                logger.warning("数据库连接已断开，尝试重新连接")
                if self.db.connect():
                    logger.info("数据库连接成功")
                else:
                    logger.error("数据库连接失败")
            return func(*args, **kwargs)
        return wrapper


if __name__ == '__main__':
    db = PostgresConnection()
    face = FaceVectorModel(label="test", vector=[0.1]*512, count=1)
    db.insert_one(face)
    item = db.find_most_similar(vector=[0.1]*512)
    print(item)