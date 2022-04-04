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

def save_my_model(trainer, gis_join):
    # SAVING TRAINED MODEL
    #trainer = DeepModel(5, 15)
    ser_model = pickle.dumps(trainer)
    info = sustain_collection.update_one({query_field:"MYGIS"}, { "$set":{query_field:gis_join, "model_obj": ser_model}}, upsert=True)
    print("FINISHED:", info)


def fetch_trained_model(gis_join):
    # FETCHING A MODEL BY ITS GISJOIN
    client_query = {query_field: gis_join}
    query_results = list(sustain_collection.find(client_query))

    pickled_model = (list(query_results))[0]
    actual_model = pickle.loads(pickled_model["model_obj"])
    return actual_model

my_model = fetch_trained_model("MYGIS")
print(my_model)

