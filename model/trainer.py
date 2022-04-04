import os
from random import sample
from time import time
from utilz.helper import fancy_logging
import pandas as pd
import pymongo
import torch
from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader

from data.MongoData import MongoData
from model.DeepNNTrainer import DeepModel
import numpy as np
import os


# SAVES A TRAINED MODEL
# SAVED MODEL NAME: net_gisjoin
def save_network(network, gis_code, save_dir):
    network = network.module if isinstance(
        network, torch.nn.DataParallel) else network
    # save_filename = '%s_net_%s_x%s.pth' % (epoch_label, network_label, scale)
    save_filename = 'net_%s.pth' % (gis_code)
    save_path = os.path.join(save_dir, save_filename)
    to_save = {
        'state_dict': network.state_dict(),
        'path': save_path
    }

    torch.save(to_save, save_path)

    return to_save

# LOADING A SAVED NETWORK
def load_network(network, saved_path):
    network = network.module if isinstance(
        network, torch.nn.DataParallel) else network

    loaded_state = torch.load(saved_path)['state_dict']
    loaded_param_names = set(loaded_state.keys())

    # allow loaded states to contain keys that don't exist in current model
    # by trimming these keys;
    own_state = network.state_dict()
    extra = loaded_param_names - set(own_state.keys())
    if len(extra) > 0:
        print('Dropping ' + str(extra) + ' from loaded states')
    for k in extra:
        del loaded_state[k]

    try:
        network.load_state_dict(loaded_state)
    except KeyError as e:
        print(e)
    print('RIKI: loaded network state for ' + saved_path)

# HANDLES ACTUAL MODEL TRAINING
def modeling(query_gisjoin, parent_gisjoin, transfer_learning, parentChange, oldModel, unique_id, epochs = 10, save_path ="/saved_models/", exhaustive=True):

    query_collection = "macav2"
    # mongo_url = "mongodb://lattice-100:27018/"

    mongo_url = str(os.getenv('SUSTAINDB_URL'))
    mongo_db_name = str(os.environ.get('SUSTAINDB_NAME'))

    # mongo_db_name = "sustaindb"
    query_fild = "gis_join"
    #exhaustive = True
    sample_percent = 0.25
    train_test_split = 0.8


    training_labels = ["min_surface_downwelling_shortwave_flux_in_air", "max_surface_downwelling_shortwave_flux_in_air",
                       "max_specific_humidity", "min_max_air_temperature", "max_max_air_temperature"]
    target_labels = ["max_min_air_temperature"]

    batch_size = 4
    n_cpu = 4

    cuda = torch.cuda.is_available()

    if not unique_id.startswith("TYPE1"):
        cuda = False

    # ACTUAL QUERYING
    sustainclient = pymongo.MongoClient(mongo_url)
    sustain_db = sustainclient[mongo_db_name]

    sustain_collection = sustain_db[query_collection]
    client_query = {query_fild: query_gisjoin}

    # creating projection
    client_projection = {}
    for val in training_labels:
        client_projection[val] = 1
    for val in target_labels:
        client_projection[val] = 1

    start_time = time()
    query_results = list(sustain_collection.find(client_query, client_projection))

    fetch_time = time() - start_time

    result_size = len(query_results)

    if exhaustive:
        all_data = query_results
    else:
        data_size = int(len(query_results) * sample_percent)
        #print("SAMPLED DATA SIZE:", data_size)
        all_data = sample(query_results, data_size)

    msk = np.random.rand(len(all_data)) < train_test_split

    all_data = pd.DataFrame(all_data)
    training_data = all_data[msk]
    val_data = all_data[~msk]

    # TRAINING DATALOADER
    X = training_data[training_labels]
    Y = training_data[target_labels]

    fancy_logging("FETCHED/ TRAINING/ VAL DATA SIZE: %s/ %s/ %s  FETCH TIME: %s secs" % (str(result_size), str(training_data.shape[0]), str(val_data.shape[0]), str(fetch_time)), unique_id)

    training_dataset = MongoData(X, Y)

    dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    # VALIDATION DATALOADER
    X1 = val_data[training_labels]
    Y1 = val_data[target_labels]

    val_dataset = MongoData(X1, Y1)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    epoch = 0

    start_time = time()
    if transfer_learning:
        # IF NEW MODEL NEEDS TO BE LOADED
        if parentChange:
            # LOAD SAVED MODEL
            trainer = DeepModel(5, 15)
            save_filename = 'net_%s.pth' % (parent_gisjoin)

            if os.path.isfile(save_path+save_filename):
                load_network(trainer,save_path+save_filename)
                fancy_logging('LOADED N/W FROM %s' % (str(save_path + save_filename)), unique_id)
            else:
                fancy_logging("!!!!!! WARNING FILE %s NOT FOUND, DOING BASIC LEARNING "%(save_path+save_filename), unique_id)

        else:
            trainer = oldModel
    else:
        # CREATE MODEL FROM SCRATCH
        trainer = DeepModel(5, 15)

    model_init_time = time() - start_time
    fancy_logging('BEGIN!!!! MODELLING FOR GIS_JOIN: %s CUDA: %s TL: %s P_CHANGE: %s MODEL_INIT_TIME: %s' % (query_gisjoin,
                                                                                                         str(cuda), str(transfer_learning),
                                                                                                         str(parentChange), str(model_init_time)), unique_id)

    if cuda:
        trainer = trainer.cuda()

    # ***************** ACTUAL TRAINING**********************************
    # for epoch in range(0, args.train.epochs):
    lowest_loss = 9999999.0
    dragging_count = 0

    early_stopped = False
    start_time = time()

    while epoch < epochs:
        criterion = nn.MSELoss()

        # Optimizer
        optimizer = optim.Adam(trainer.parameters(), weight_decay=0.0001)

        trainer.train()

        train_losses = []
        valid_losses = []

        # TRAINING********************************
        for i, record in enumerate(dataloader):
            # print("ACT",record.size())
            if cuda:
                record['X'] = record['X'].cuda()
                record['Y'] = record['Y'].cuda()

            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = trainer(record['X'].float())

            loss = criterion(outputs, record['Y'].float())
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            #if i % 200 == 0:
                #print(epoch, i, loss.item())

        # EVALUATION********************************
        trainer.eval()
        for i, record in enumerate(val_dataloader):
            # print("ACT",record.size())
            if cuda:
                record['X'] = record['X'].cuda()
                record['Y'] = record['Y'].cuda()

            # Forward + backward + optimize
            outputs = trainer(record['X'].float())

            #print(type(outputs), record['Y'].size())
            loss = criterion(outputs, record['Y'].float())

            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        if valid_loss < lowest_loss:
            dragging_count = 0
            lowest_loss = valid_loss
        else:
            dragging_count += 1

        if dragging_count > 2:
            fancy_logging("LOSSES TRAIN %s VAL %s PREV_VAL %s DECISION: STOP" % (str(train_loss), str(valid_loss), str(lowest_loss)), hostname=unique_id)
            early_stopped = True
            break
        else:
            fancy_logging("LOSSES TRAIN %s VAL %s PREV_VAL %s DECISION: CONTINUE" % (str(train_loss), str(valid_loss), str(lowest_loss)), hostname=unique_id)

        epoch += 1

    train_time = time() - start_time
    fancy_logging("FINISHED!!!! EPOCHS: %s EARLY STOPPAGE?? %s, TRAIN_TIME: %s"%(str(epoch), str(early_stopped), str(train_time)), unique_id)

    fancy_logging("SAVING MODEL AT: %s" % (str(save_path)), unique_id)
    save_network(trainer, query_gisjoin, save_path)
    fancy_logging("SAVED MODEL AT: %s" % (str(save_path)), unique_id)
    return trainer