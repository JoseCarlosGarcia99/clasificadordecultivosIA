from flask_pymongo import ObjectId
from bson.json_util import dumps

def deleteUser(db,id):
    result = db.users.delete_one({"_id":ObjectId(id)})
    return result