import os
import os.path as osp
import argparse
from utilz.helper import fancy_logging
from pytorch_lightning import Trainer
import socket
import logging
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from data.mongo_model_utils import *
import shutil
import pickle
import torch
from time import time
from model.Trainer_AE_Light import AETrainerLight
from model.Trainer_IP_Light import IPTrainerLight

hostname=str(socket.gethostname())
base_path="/s/"+hostname+"/a/nobackup/galileo/sapmitra/SRImages"
# HOW OFTEN IN AN EPOCH TO VALIDATE

configFile="/s/chopin/b/grad/sapmitra/Glance_master/config/prosrgan.yaml" #Config File
ig_file="/s/chopin/b/grad/sapmitra/Glance_master/config/ignorables.txt"
n_cpu=8 #number of cpu threads to use during batch generation
file_separator = "/"

def clear_or_create(p):
    if not osp.isdir(p):
        os.makedirs(p)
    else:
        # REMOVE ENTRIES
        for f in os.listdir(p):
            if not osp.isdir(os.path.join(p, f)):
                os.remove(os.path.join(p, f))
            else:
                shutil.rmtree(os.path.join(p, f))

def clear_old_log(old_log):
    if os.path.exists(old_log):
        print("REMOVED OLD LOG")
        os.remove(old_log)


def execute_ml(quad_to_train, parent_quad, my_hostname, sample_percent, num_epochs, training_type = 1, my_type ="ANCHOR", num_nodes = 15, logger=None):
    #num_nodes = 2
    validation_freq = 0.5
    flush_steps = 100
    shutdown_patince = 24

    start_time = time()
    fancy_logging("MODEL INIT FOR GISJOIN %s TIME TAKEN: %s" % (quad_to_train, str(time() - start_time)), my_hostname,
                  logger)

    early_stop = EarlyStopping(monitor='val_loss', patience = shutdown_patince, strict=False, verbose=True, mode='min')
    # CHANGE THIS TO LOCAL CODE
    if training_type == 1:
        trainer_model = AETrainerLight(sample_percent)
    else:
        trainer_model = IPTrainerLight(sample_percent)

    trainer = Trainer(gpus=1, num_nodes = num_nodes, max_epochs=50, min_epochs=24, distributed_backend='ddp')

    '''trainer = Trainer(gpus=1, val_check_interval=validation_freq, flush_logs_every_n_steps=flush_steps, callbacks=[early_stop],
                      max_epochs=num_epochs, min_epochs=4)'''

    trainer.fit(trainer_model)
    # SAVING TRAINED MODEL IN-MEMORY AND IN MONGO-DB
    if "lattice-181" in hostname:
        print("*****************SAVING GLOBAL MODEL AT ", hostname)
        save_model(trainer_model.model, "GLOBAL", {}, trainer_model.val_loss1, training_type, isTL=True)
    fancy_logging("MODEL SAVE FOR GISJOIN %s TIME TAKEN: %s" % (quad_to_train, str(time() - start_time)), my_hostname, logger)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Individual ML Trainer Module')
    parser.add_argument('-sp', '--save_path', help='Save Path for Trained Parents', required=False, default="/saved_models/")
    parser.add_argument('-hs', '--host_server', help='Coordinator Machine Name', required=False, default="129.82.45.188")
    parser.add_argument('-hp', '--host_port', help='Port for The Coordinator Machine', required=False, type=int, default=31477)
    parser.add_argument('-e', '--exhaustive', help='Is Exhaustive?', required=False, type=str, default="False")
    parser.add_argument('-non_exhaustive', dest='feature', action='store_false')
    parser.add_argument('-nep', '--epochs', help='Port for The Coordinator Machine', required=False, type=int, default=100)
    parser.add_argument('-m', '--mode', help='Type of Model: 1 = AE; 2 = Inpainter', required=False, type=int, default=2)
    parser.add_argument('-p', '--samperc', help='Child Sample Percent', required=False, type=float, default=0.1)
    parser.add_argument('-cl', '--clustersize', help='Size of the Cluster', required=False, type=int, default=15)
    parser.set_defaults(feature=True)

    # USING A UNIQUE ID FOR A CLIENT
    my_hostname = str(socket.gethostname())
    args = vars(parser.parse_args())
    # TELLS US WHICH MODEL TO TRAIN
    ae_or_clas = args['mode']
    num_nodes = int(args['clustersize'])

    print("TRAINING MODE: ", ae_or_clas)
    # PATH WHERE TRAINED MODELS WILL BE SAVED
    base_path = "/s/" + my_hostname + "/a/nobackup/galileo/sapmitra/"
    model_save_path = base_path + "tl_models"
    log_file_name = base_path + 'tl_global_' + my_hostname + "_" + str(ae_or_clas) + '.log'
    clear_old_log(log_file_name)

    logger = logging.getLogger("tl")
    logging.basicConfig(filename=log_file_name, level=logging.INFO)

    # IF PATH DOES NOT EXIST, OR REMOVE INNER CONTENTS
    clear_or_create(model_save_path)

    host = args['host_server']
    port = args['host_port']
    num_epochs = args['epochs']
    child_sp = float(args['samperc'])

    container_type = str(os.getenv('CONT_TYPE'))

    cuda = torch.cuda.is_available()

    fancy_logging("LET US BEGIN!!!!!!!!!!!", my_hostname, logger)
    fancy_logging("UNIQUE CLIENT ID IS: %s CUDA: %s" % (my_hostname, str(cuda)), my_hostname, logger)
    fancy_logging("LOG PATH: " + log_file_name, my_hostname, logger)
    i = 0

    sample_percent = child_sp
    fancy_logging("TRAINING PARENTS..............", my_hostname, logger)
    start_time = time()

    my_type = "ANCHOR"

    print("!!!!!!!!num_nodes!!!!!!!!", num_nodes)
    execute_ml("", 'default', my_hostname, sample_percent, num_epochs, ae_or_clas, my_type, num_nodes, logger)


    fancy_logging("TRAINING ALL PARENTS DONE.............." + str(time() - start_time), my_hostname, logger)

    print("\n\n\nTOTAL TIME:", str(time() - start_time),"\n\n\n")

