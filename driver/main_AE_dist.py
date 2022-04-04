import os
import os.path as osp
import sys
import yaml
from utilz.dotdict import DotDict
from pytorch_lightning import Trainer
import socket
import logging
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from model.trainer_dist_ae import AETrainer
import shutil
from time import time


hostname=str(socket.gethostname())
base_path="/s/"+hostname+"/a/nobackup/galileo/sapmitra/SRImages"
# HOW OFTEN IN AN EPOCH TO VALIDATE
validation_freq = 0.25

configFile="config/ae.yaml" #Config File
file_separator = "/"

def clear_old(params):
    old_log = base_path + '/tl_light_' + hostname + '.log'
    if os.path.exists(old_log):
        print("REMOVED OLD LOG")
        os.remove(old_log)
    clear_or_create(base_path + params.xtra.save_path)


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

def modelling(args):

    num_nodes = 2
    clear_old(args)
    clear_or_create(base_path+args.xtra.chkpt_path)

    ignorables = []
    args.ignorables = ignorables

    logging.basicConfig(filename=base_path + '/tl_light_' + hostname + '.log', level=logging.INFO)
    print("BASE PATH: " + base_path)
    print("LOG PATH:", (base_path + '/tl_light_' + hostname + '.log'))

    # checkpoint is the directory where checkpoints are read from and stored
    trainer_model = AETrainer(opt=args, base_dir=base_path, save_dir=base_path+args.xtra.chkpt_path)

    #********************** TRAINING x2 ********************************
    early_stop = EarlyStopping(monitor='val_loss', patience = args.train.training_shutdown_patience, strict=False, verbose=True, mode='min')
    #chk_name = 'glance-{epoch:02d}-{val_loss:.2f}'
    chk_name = 'tl_light-ae'
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath=base_path+args.xtra.chkpt_path, filename=chk_name, mode='min')
    #trainer = Trainer(gpus=1,num_nodes=2, max_epochs=args.train.epochs, distributed_backend='ddp')
    '''trainer = Trainer(gpus=1, num_nodes=num_nodes, val_check_interval=validation_freq, callbacks=[early_stop, checkpoint_callback],
                      max_epochs=args.train.max_epochs,
                      min_epochs=args.train.min_epochs, distributed_backend='ddp')'''
    trainer = Trainer(gpus=1, val_check_interval=validation_freq, callbacks=[early_stop, checkpoint_callback],
                      max_epochs=args.train.max_epochs, min_epochs=args.train.min_epochs)
    start_time = time()
    trainer.fit(trainer_model)
    logging.info("**********************************DONE**********************TOTAL TRAIN TIME: "+ str(time() - start_time))
    print("*******************************RIKI: FINISHED TRAINING...SLEEPING")
    trainer_model.save(0,0,0,True)



if __name__ == '__main__':

    with open(configFile) as file:
        try:
            params = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(0)

    params = DotDict(params)

    #pprint(params)
    modelling(params)
