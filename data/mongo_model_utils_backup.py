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

# SAVING TRAINED MODEL IN MONGODB
# ALSO SAVES IT IN IN-MEMORY MAP
def save_model(trained_model, my_quad, trained_model_dict, model_err, isTL = False):
    # LOCALLY SAVE IN-MEMORY ONLY FOR CENTROID MODELS
    # SAVE OTHERS IN MONGO-DB
    model_infos = {"model_obj": trained_model, "model_err":model_err}
    if not isTL:
        trained_model_dict[my_quad] = model_infos
    ser_model = pickle.dumps(trained_model)
    info = sustain_collection.update_one({query_field:my_quad}, { "$set":{query_field:my_quad, "model_obj": ser_model, "model_err": model_err}}, upsert=True)
    print("SAVED TRAINED_MODEL:", info)


# FETCHING A CENTROID MODEL BY ITS GISJOIN FROM MEMORY OR MONGODB
def fetch_model_MONGO_MEMORY(my_quad, trained_model_dict):
    # THE TRAINED MODEL IS ALREADY IN-MEMORY
    if my_quad in trained_model_dict:
        return trained_model_dict[my_quad]["model_obj"], trained_model_dict[my_quad]["model_err"]
    else:
    # NOT PRESENT LOCALLY, FETCH FROM MONGODB AND POPULATE TRAINED DICTIONARY
        print("!!!!!!!!!!!!!!!NOT PRESENT LOCALLY....FETCHING FROM MONGO!!!!!!!!!!!!!!!!")
        client_query = {query_field: my_quad}
        query_results = list(sustain_collection.find(client_query))

        query_results = list(query_results)
        if len(query_results) > 0:
            trained_model = pickle.loads(query_results[0]["model_obj"])
            model_err = query_results[0]["model_err"]
            trained_model_dict[my_quad] = {"model_obj": trained_model, "model_err":model_err}
            return trained_model, model_err
        else:
            # THIS MODEL DOES NOT EXIST LOCALLY OR IN MONGODB
            return None, None
'''
old_model = DeepModel(5, 15)
save_model(old_model, "MYGIS", {})

trained_dict = {}
fetch_model_MONGO_MEMORY("MYGIS", trained_dict)
print(trained_dict)

'''
