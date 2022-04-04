import pymongo
from data.MongoData import MongoData
from torch.utils.data import DataLoader
#from torchvision import transforms
from model.DeepNNTrainer import DeepModel
from random import sample
import pandas as pd
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
import my_que.my_queue_reader as mq

file_separator = "/"

def modelling(query_gisjoin):
    n_cpu = 0  # number of cpu threads to use during batch generation
    # QUERY PARAMETERS
    #query_gisjoin = "G4601370"
    query_collection = "macav2"
    #mongo_url = "mongodb://lattice-100:27018/"

    mongo_url = str(os.getenv('SUSTAINDB_URL'))
    mongo_db_name = str(os.environ.get('SUSTAINDB_NAME'))

    #mongo_db_name = "sustaindb"
    query_fild = "gis_join"
    exhaustive = False
    sample_percent = 0.25
    training_labels = ["min_surface_downwelling_shortwave_flux_in_air","max_surface_downwelling_shortwave_flux_in_air",
                       "max_specific_humidity", "min_max_air_temperature", "max_max_air_temperature"]
    target_labels = ["max_min_air_temperature"]
    epochs = 10
    batch_size = 1

    cuda = torch.cuda.is_available()

    print("MODELLING FOR GIS_JOIN:", query_gisjoin, cuda)

    if cuda:
        n_cpu=1

    print("\n====================\nCUDA:", cuda, "\n=====================\n")
    cond = True
    if cond:
        return

    # ACTUAL QUERYING
    sustainclient = pymongo.MongoClient(mongo_url)
    sustain_db = sustainclient[mongo_db_name]

    sustain_collection = sustain_db[query_collection]
    client_query = {query_fild: query_gisjoin}

    #creating projection
    client_projection = {}
    for val in training_labels:
        client_projection[val] = 1
    for val in target_labels:
        client_projection[val] = 1

    #print(client_projection)
    query_results = list(sustain_collection.find(client_query, client_projection))

    result_size = len(query_results)
    print("FETCHED DATA SIZE: ", result_size)

    if exhaustive:
        training_data = query_results
    else:
        data_size = int(len(query_results) * sample_percent)
        #print("SAMPLED DATA SIZE:", data_size)
        training_data = sample(query_results, data_size)

    training_data = pd.DataFrame(training_data)

    X = training_data[training_labels]
    Y = training_data[target_labels]

    print("TRAINING DATA SIZE:", training_data.shape[0])
    training_dataset = MongoData(X,Y)

    dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    epoch = 0
    # for epoch in range(0, args.train.epochs):
    while epoch < epochs:
        trainer = DeepModel(5, 15)
        iter_start_time = time()
        if cuda:
            trainer = trainer.cuda()

        criterion = nn.MSELoss()

        # Optimizer
        optimizer = optim.Adam(trainer.parameters(), weight_decay=0.0001)

        for i, record in enumerate(dataloader):
            #print("ACT",record.size())
            if cuda:
                record['X'] = record['X'].cuda()
                record['Y'] = record['Y'].cuda()

            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = trainer(record['X'].float())
            loss = criterion(outputs, record['Y'].float())
            loss.backward()
            optimizer.step()

            if i%200 == 0:
                print(epoch, i, loss.item())
        epoch+=1

if __name__ == '__main1__':

    redis_host = "redis"
    q = mq.RedisWQ(name="gis_q", host=redis_host)

    print("Worker with sessionID: " + q.sessionID())
    print("Initial my_que state: empty=" + str(q.empty()))

    i = 0
    while True:
        if not q.empty():
            # lease_secs : expiry time after which some other container
            # can pick up the object
            item = q.lease(lease_secs=300, block=False)
            if item is not None:
                gis_code = item.decode("utf-8")
                print("=====================================")
                print("PRE-MODELLING FOR THE GIS_JOIN: ", gis_code) # Put your actual work here instead of sleep.
                modelling(gis_code)
                q.complete(item)
                print("RELEASED THE GIS_JOIN: ", gis_code)
                print("=====================================")
            else:
                print("Waiting for work")
        else:
            i+=1
            if i%500 == 0:
                print("EMPTY QUEUE...NOTHING TO DO")

