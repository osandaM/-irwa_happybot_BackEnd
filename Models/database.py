# Import the pymongo library for MongoDB interaction and import the 'config' module.
import pymongo
from config import config

# Function to connect to MongoDB.
def connect_to_mongodb():
    # Create a MongoDB client using the URI from the 'config' module.
    client = pymongo.MongoClient(config.MONGO_URI)
    # Return the database specified in the 'config' module.
    return client[config.MONGO_DB_NAME]

# Function to insert data into a specified collection.
def insert_data(collection_name, data):
    # Connect to MongoDB.
    db = connect_to_mongodb()
    # Access the specified collection.
    collection = db[collection_name]
    # Insert the provided data into the collection and capture the result.
    result = collection.insert_one(data)
    # Return the result of the insertion operation.
    return result

# Function to retrieve data from a specified collection with an optional query.
def retrieve_data(collection_name, query={}):
    # Connect to MongoDB.
    db = connect_to_mongodb()
    # Access the specified collection.
    collection = db[collection_name]
    # Retrieve data from the collection based on the provided query (default is an empty query).
    result = collection.find(query)
    # Return the result of the retrieval operation.
    return result
