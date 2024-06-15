from flask_pymongo import ObjectId
from bson.json_util import dumps

def getUser(db,id):
    user = db.users.find_one(ObjectId(id))
    response = dumps(user)
    return response