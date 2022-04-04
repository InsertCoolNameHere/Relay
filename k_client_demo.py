import os
from data.NOA_NAM_Data import NOAA_Data
from torch.utils.data import DataLoader
from data.mongo_datafetch_utils import training_labels
from model.DeepNNTrainer import DeepModel
from time import time
import torch
import torch.nn as nn
import torch.optim as optim

file_separator = "/"

def modeling(query_gisjoin):
    n_cpu = 0  # number of cpu threads to use during batch generation
    epochs = 1
    batch_size = 1
    model_save_path = "/tmp"

    cuda = torch.cuda.is_available()

    print("MODELLING FOR GIS_JOIN:", query_gisjoin, cuda)

    if cuda:
        n_cpu=1

    print("\n====================\nCUDA:", cuda, "\n=====================\n")

    training_dataset = NOAA_Data(query_gisjoin)

    dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    print("BEGIN TRAINING.............")
    epoch = 0
    # for epoch in range(0, args.train.epochs):
    while epoch < epochs:
        trainer = DeepModel(len(training_labels), 15)
        iter_start_time = time()
        if cuda:
            trainer = trainer.cuda()

        criterion = nn.MSELoss()

        # Optimizer
        optimizer = optim.Adam(trainer.parameters(), weight_decay=0.0001)

        for i, record in enumerate(dataloader):
            record['X'] = record['X'].unsqueeze(0)
            if cuda:
                record['X'] = record['X'].cuda()
                record['Y'] = record['Y'].cuda()

            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = trainer(record['X'].float())

            loss = criterion(outputs, record['Y'].float())
            loss.backward()
            optimizer.step()

            if i%1000 == 0:
                print(epoch, i, loss.item())
        epoch+=1

    # ************** SAVING TRAINED MODEL ******************

    c_time = str(time())
    save_filename = 'net_%s.pth' % (c_time)
    save_path = os.path.join(model_save_path, save_filename)
    save_model(trainer, save_path)

    # ************** LOADING TRAINED MODEL ******************
    print("--------------------------")
    trainer1 = DeepModel(len(training_labels), 15)
    load_model(trainer1, save_path)

def save_model(model, save_path):
    to_save = {
        'state_dict': model.state_dict(),
    }
    torch.save(to_save, save_path)

    # PRINT SAVED PARAMETERS
    '''for param in model.parameters():
        print(param.data)'''

def load_model(model, save_path):
    loaded_state = torch.load(save_path)['state_dict']
    loaded_param_names = set(loaded_state.keys())
    own_state = model.state_dict()
    extra = loaded_param_names - set(own_state.keys())
    if len(extra) > 0:
        print('Dropping ' + str(extra) + ' from loaded states')
    for k in extra:
        del loaded_state[k]
    model.load_state_dict(loaded_state)

    # PRINT LOADED PARAMETERS
    '''for param in model.parameters():
        print(param.data)'''


if __name__ == '__main__':
    # QUERY PARAMETERS
    query_gisjoin = "G0100270"
    modeling(query_gisjoin)
