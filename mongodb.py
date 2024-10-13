
from pymongo.mongo_client import MongoClient
import urllib.parse



# insert a doc

def insert_into_db(num_people):
    
    # Example username and password
    username = "test"
    password = "abcd"  # Contains special character '@' and '!' which need encoding

    # URL-encode the username and password
    encoded_username = urllib.parse.quote_plus(username)
    encoded_password = urllib.parse.quote_plus(password)

    # Now use the encoded values in the connection string
    uri = f"mongodb+srv://{encoded_username}:{encoded_password}@project.s4o1r.mongodb.net/?retryWrites=true&w=majority&appName=Project"

    print(uri)


    #uri = "mongodb+srv://<test@admin>:<abcd>@project.s4o1r.mongodb.net/?retryWrites=true&w=majority&appName=Project"

    # Create a new client and connect to the server
    client = MongoClient(uri)




    # Providing database and collection

    db = client["Drone"]
    collection = db["People"]

    mydocument = {  
        "num_people": num_people,
    }

    insert_doc = collection.insert_one(mydocument)
    print(f"Doc insert succefull. The doc id is{insert_doc.inserted_id}")
    


    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    client.close


# close connection


