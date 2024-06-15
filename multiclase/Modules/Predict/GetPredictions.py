from bson.json_util import dumps

def getUsers(db):
    users = db.users.find()
    response = dumps(users)
    return response