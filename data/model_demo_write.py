from model.DeepNNTrainer import DeepModel
import pymongo
import pickle

# MONGO_SETUP
query_collection = "SavedModels"
mongo_url = "mongodb://lattice-100:27018/"
mongo_db_name = "sustaindb"
query_field = "gis_join"
sustainclient = pymongo.MongoClient(mongo_url)
sustain_db = sustainclient[mongo_db_name]
sustain_collection = sustain_db[query_collection]

# SAVING TRAINED MODEL
trainer = DeepModel(5, 15)
ser_model = pickle.dumps(trainer)
info = sustain_collection.update_one({query_field:"MYGIS"}, { "$set":{query_field:"MYGIS", "model_obj": ser_model}}, upsert=True)
print("FINISHED:", info)


# FETCHING A MODEL BY ITS GISJOIN

client_query = {query_field: "MYGIS"}

query_results = list(sustain_collection.find(client_query))

print(list(query_results))

