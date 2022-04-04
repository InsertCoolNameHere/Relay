from model.DeepNNTrainer import DeepModel
import pymongo
import pickle
import gridfs

# SAMPLE NOAA_NAM DATA:
'''{
	"_id" : ObjectId("61ef78efe3b87d5fce8c0636"),
	"ROW" : 148,
	"COL" : 417,
	"LATITUDE" : 33.44567725782628,
	"LONGITUDE" : -85.8535273727299,
	"DATE" : ISODate("2021-02-01T00:00:00Z"),
	"TIMESTEP_HOURS" : "000",
	"TIMESTAMP_MS_SINCE_EPOCH" : 1612162800000,
	"PRESSURE_REDUCED_TO_MSL_PASCAL" : 100939.28,
	"VISIBILITY_AT_SURFACE_METERS" : 24100,
	"VISIBILITY_AT_CLOUD_TOP_METERS" : 15300,
	"U_COMPONENT_OF_WIND_AT_PLANETARY_BOUNDARY_LAYER_METERS_PER_SEC" : 6.930843353271484,
	"V_COMPONENT_OF_WIND_AT_PLANETARY_BOUNDARY_LAYER_METERS_PER_SEC" : -3.2463130950927734,
	"WIND_GUST_SPEED_AT_SURFACE_METERS_PER_SEC" : 8.754495239257812,
	"PRESSURE_AT_SURFACE_PASCAL" : 97452.55625000001,
	"TEMPERATURE_AT_SURFACE_KELVIN" : 285.923046875,
	"SOIL_TEMPERATURE_0_TO_01_M_BELOW_SURFACE_KELVIN" : 286.91,
	"VOLUMETRIC_SOIL_MOISTURE_CONTENT_0_TO_01_M_BELOW_SURFACE_FRACTION" : 0.35100000000000003,
	"SNOW_COVER_AT_SURFACE_PERCENT" : 0,
	"SNOW_DEPTH_AT_SURFACE_METERS" : 0,
	"TEMPERATURE_2_METERS_ABOVE_SURFACE_KELVIN" : 287.84904296875,
	"DEWPOINT_TEMPERATURE_2_METERS_ABOVE_SURFACE_KELVIN" : 282.38615234375,
	"RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT" : 69.98242492675782,
	"U_COMPONENT_OF_WIND_10_METERS_ABOVE_SURFACE_METERS_PER_SEC" : 3.72382568359375,
	"V_COMPONENT_OF_WIND_10_METERS_ABOVE_SURFACE_METERS_PER_SEC" : -1.6467541503906251,
	"TOTAL_PRECIPITATION_SURFACE_ACCUM_KG_PER_SQ_METER" : 0,
	"CONVECTIVE_PRECIPITATION_SURFACE_ACCUM_KG_PER_SQ_METER" : 0,
	"GEOPOTENTIAL_HEIGHT_LOWEST_LEVEL_WET_BULB_ZERO_GPM" : 2294.4326171875,
	"CATEGORICAL_SNOW_SURFACE_BINARY" : 0,
	"CATEGORICAL_ICE_PELLETS_SURFACE_BINARY" : 0,
	"CATEGORICAL_FREEZING_RAIN_SURFACE_BINARY" : 0,
	"CATEGORICAL_RAIN_SURFACE_BINARY" : 1,
	"VEGETATION_SURFACE_PERCENT" : 42.1,
	"VEGETATION_TYPE_SIB_INT" : 5,
	"SOIL_TYPE_ZOBLER_INT" : 4,
	"ALBEDO_PERCENT" : 15,
	"TOTAL_CLOUD_COVER_PERCENT" : 95,
	"GISJOIN" : "G0100270"
}'''

# MONGO_SETUP
query_collection = "noaa_nam"
mongo_url = "mongodb://lattice-101:27018/"
mongo_db_name = "sustaindb"
query_field = "GISJOIN"
type_field = "model_type"

training_labels = ["PRESSURE_REDUCED_TO_MSL_PASCAL" ,"VISIBILITY_AT_SURFACE_METERS" ,"VISIBILITY_AT_CLOUD_TOP_METERS" ,
                   "U_COMPONENT_OF_WIND_AT_PLANETARY_BOUNDARY_LAYER_METERS_PER_SEC" ,"V_COMPONENT_OF_WIND_AT_PLANETARY_BOUNDARY_LAYER_METERS_PER_SEC" ,
                   "WIND_GUST_SPEED_AT_SURFACE_METERS_PER_SEC" ,"PRESSURE_AT_SURFACE_PASCAL" ,"TEMPERATURE_AT_SURFACE_KELVIN"]
target_labels = ["SOIL_TEMPERATURE_0_TO_01_M_BELOW_SURFACE_KELVIN"]

sustainclient = pymongo.MongoClient(mongo_url)
sustain_db = sustainclient[mongo_db_name]
fs = gridfs.GridFS(sustain_db)
sustain_collection = sustain_db[query_collection]


to_fetch = []
to_fetch.extend(training_labels)
to_fetch.extend(target_labels)

fetch_dictionary = {"_id":0}
for lab in to_fetch:
    fetch_dictionary[lab] = 1


# FETCHING A CENTROID MODEL BY ITS GISJOIN FROM MEMORY OR MONGODB
def fetch_training_data_MONGO(geo_identifier):
    #client_query = {"$and": [{query_field : geo_identifier}, {type_field: model_type}]}
    client_query = {query_field: geo_identifier}
    query_results = sustain_collection.find(client_query,fetch_dictionary)

    query_results = list(query_results)
    return query_results


fetch_training_data_MONGO("G0100270")