from model.DeepNNTrainer import DeepModel
import pymongo
import pickle
import gridfs

# MONGO_SETUP
query_collection = "SavedModels"
mongo_url = "mongodb://lattice-101:27018/"
mongo_db_name = "sustaindb"
query_field = "gis_join"
type_field = "model_type"
sustainclient = pymongo.MongoClient(mongo_url)
sustain_db = sustainclient[mongo_db_name]
fs = gridfs.GridFS(sustain_db)
sustain_collection = sustain_db[query_collection]

# SAVING TRAINED MODEL IN MONGODB
# ALSO SAVES IT IN IN-MEMORY MAP
def save_model(trained_model, my_quad, trained_model_dict, model_err, model_type, isTL = False):
    # LOCALLY SAVE IN-MEMORY ONLY FOR CENTROID MODELS
    # SAVE OTHERS IN MONGO-DB
    model_infos = {"model_obj": trained_model, "model_err":model_err}
    if not isTL:
        #x=1
        trained_model_dict[my_quad] = model_infos
    ser_model = pickle.dumps(trained_model)
    model_key = fs.put(ser_model)

    #print("SAVED KEY: ", model_key)

    info = sustain_collection.update_one({"$and": [{query_field : my_quad},{type_field: model_type}]} , { "$set":{query_field:my_quad, "model_key": model_key, type_field: model_type, "model_err": model_err}}, upsert=True)
    print("SAVED TRAINED_MODEL:", info)


# FETCHING A CENTROID MODEL BY ITS GISJOIN FROM MEMORY OR MONGODB
def fetch_model_MONGO_MEMORY(my_quad, trained_model_dict, model_type):
    # THE TRAINED MODEL IS ALREADY IN-MEMORY
    if my_quad in trained_model_dict:
        return trained_model_dict[my_quad]["model_obj"], trained_model_dict[my_quad]["model_err"]
    else:
    # NOT PRESENT LOCALLY, FETCH FROM MONGODB AND POPULATE TRAINED DICTIONARY
        print("!!!!!!!!!!!!!!!NOT PRESENT LOCALLY....FETCHING FROM MONGO!!!!!!!!!!!!!!!!")
        client_query = {"$and": [{query_field : my_quad},{type_field: model_type}]}
        query_results = list(sustain_collection.find(client_query))

        query_results = list(query_results)
        if len(query_results) > 0:
            trained_model_key = query_results[0]["model_key"]
            print("RETRIEVED TYPE: ", query_results[0][type_field])
            trained_model = pickle.loads(fs.get(trained_model_key).read())

            #print(trained_model.state_dict())
            model_err = query_results[0]["model_err"]
            trained_model_dict[my_quad] = {"model_obj": trained_model, "model_err":model_err}
            return trained_model, model_err
        else:
            # THIS MODEL DOES NOT EXIST LOCALLY OR IN MONGODB
            print("MODEL IS A GHOST OOOOOOOOOOOOOOO")
            return None, None
'''
old_model = DeepModel(5, 15)
save_model(old_model, "MYGIS", {})

trained_dict = {}
fetch_model_MONGO_MEMORY("MYGIS", trained_dict)
print(trained_dict)

'''
