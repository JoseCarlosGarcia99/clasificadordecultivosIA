from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def databaseConnection(app): 
    uri = "mongodb+srv://josecarlosgarciavelez99:Cronos123@cluster0.7lnvn59.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri, server_api=ServerApi('1'))
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Connected to Atlas")
    except Exception as e:
        print(e)
    db = client.clasificaciondecultivos

    return db