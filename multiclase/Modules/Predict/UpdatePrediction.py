from flask_pymongo import ObjectId

def updateUser(db, request, id):
    query = {"_id":ObjectId(id)}
    newvalues = { "$set": request.json}
    result = db.users.update_one(query, newvalues)
    return result