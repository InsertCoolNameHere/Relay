import logging
import argparse
from cluster.detect_cluster import cluster_setup
import os
import os.path as osp
from utilz.helper import fancy_logging
# import my_que.my_queue_reader as mq
from model.trainer_ae import *
from model.trainer_ip import *
from model.DeepNNTrainer import DeepModel
from time import time,sleep
import torch
import shutil
from data.mongo_model_utils import *


file_separator = "/"
node_qhash_map = {}
qhash_node_map = {}
parent_inner_qhash_map = {}
node_centroid_map = {}

trained_models = {}

# CREATE MODEL FROM SCRATCH
def from_scratch(mode):
    if mode == 1:
        # THIS IS FOR AUTOENCODER
        return get_base_model_AE()
    else:
        # THIS IS FOR THE CLASSIFIER
        return get_base_model_IP()

def execute_ml(quad_to_train, parent_quad, my_hostname, sample_percent, num_epochs, training_type = 1, my_type ="ANCHOR", logger=None):
    # IS THIS TRANSFER OR PARENT LEARNING?
    transferLearning = True
    if 'default' in parent_quad:
        transferLearning = False

    start_time = time()

    parent_err = 0
    # GET INITIAL MODEL OBJECT
    if transferLearning:
        # FETCH THE OLD MODEL
        base_model, parent_err = fetch_model_MONGO_MEMORY(parent_quad, trained_models, training_type)
        # IF NOTHING CAME BACK
        while not base_model:
            print("THIS GISJOIN'S PARENT HAS NOT COMPLETED TRAINING ON SOME OTHER NODE... TRYING AGAIN AFTER 5 SECS", quad_to_train, parent_quad)
            fancy_logging("THIS GISJOIN %s's PARENT HAS NOT COMPLETED TRAINING ON SOME OTHER NODE... TRYING AGAIN AFTER 5 SECS" % (
                    quad_to_train), my_hostname, logger)
            sleep(10)
            start_time = time()
            base_model, parent_err = fetch_model_MONGO_MEMORY(parent_quad, trained_models, training_type)
            # base_model = from_scratch(training_type)
            transferLearning = False
    else:
        base_model = from_scratch(training_type)

    fancy_logging("MODEL INIT FOR GISJOIN %s TIME TAKEN: %s" % (quad_to_train, str(time() - start_time)), my_hostname,
                  logger)

    # CHANGE THIS TO LOCAL CODE
    if training_type == 1:
        trained_model, val_loss, epochs = modeling_ae(quad_to_train, parent_err, transferLearning, base_model,
                                                      my_hostname,
                                                      epochs=num_epochs, sample_percent=sample_percent, logger=logger)
    else:
        trained_model, val_loss, epochs = modeling_ip(quad_to_train, parent_err, transferLearning, base_model,
                                                      my_hostname,
                                                      epochs=num_epochs, sample_percent=sample_percent, logger=logger)

    start_tim1 = time()
    # SAVING TRAINED MODEL IN-MEMORY AND IN MONGO-DB
    save_model(trained_model, quad_to_train, trained_models, val_loss, training_type, isTL=transferLearning)
    fancy_logging("MODEL SAVE FOR GISJOIN %s TIME TAKEN: %s" % (quad_to_train, str(time() - start_tim1)), my_hostname,
                  logger)

    fancy_logging('FINISHED MODELLING FOR %s C/O %s TYPE: %s TOTAL_TIME_TAKEN %s VAL_LOSS: %s PARENT_LOSS %s NUM_EPOCHS: %d'
                  % (quad_to_train, my_type, parent_quad, str(time() - start_time), str(val_loss),str(parent_err), epochs), my_hostname,
                  logger)

# MAKE A FOLDER OR CLEAR OLD CONTENTS OF THE FOLDER
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

def get_parent_quad(target_quad):
    cluster_id = target_quad[0:8]
    centroid_quad = "default"
    if cluster_id in parent_inner_qhash_map:
        centroid_quad = parent_inner_qhash_map[cluster_id]['centroid']
    return centroid_quad


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Individual ML Trainer Module')
    parser.add_argument('-sp', '--save_path', help='Save Path for Trained Parents', required=False, default="/saved_models/")
    parser.add_argument('-hs', '--host_server', help='Coordinator Machine Name', required=False, default="129.82.45.188")
    parser.add_argument('-hp', '--host_port', help='Port for The Coordinator Machine', required=False, type=int, default=31477)
    parser.add_argument('-e', '--exhaustive', help='Is Exhaustive?', required=False, type=str, default="False")
    parser.add_argument('-non_exhaustive', dest='feature', action='store_false')
    parser.add_argument('-nep', '--epochs', help='Port for The Coordinator Machine', required=False, type=int, default=200)
    parser.add_argument('-m', '--mode', help='Type of Model: 1 = AE 2 = Classifier', required=False, type=int, default=1)
    parser.add_argument('-p', '--samperc', help='Child Sample Percent', required=False, type=float, default=0.1)
    parser.set_defaults(feature=True)

    # USING A UNIQUE ID FOR A CLIENT
    my_hostname = str(socket.gethostname())
    args = vars(parser.parse_args())
    # TELLS US WHICH MODEL TO TRAIN
    ae_or_clas = args['mode']
    # PATH WHERE TRAINED MODELS WILL BE SAVED
    base_path = "/s/" + my_hostname + "/a/nobackup/galileo/sapmitra/"
    model_save_path = base_path + "tl_models"
    log_file_name = base_path + 'tl_hybrid_' + my_hostname + "_" + str(ae_or_clas) + '.log'
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
    fancy_logging("LOG PATH: "+log_file_name, my_hostname, logger)

    # INITIALIZING CLUSTER
    node_qhash_map, qhash_node_map, parent_inner_qhash_map, node_centroid_map = cluster_setup("cluster/nodes.csv")
    i=0

    sample_percent = 1
    fancy_logging("TRAINING PARENTS..............", my_hostname, logger)
    start_time = time()

    my_type = "ANCHOR"
    # **********************TRAIN THE CENTROIDS FIRST
    my_centroids = node_centroid_map[my_hostname]
    for quad_dict in my_centroids:
        quad_to_train = quad_dict['centroid']
        execute_ml(quad_to_train, 'GLOBAL', my_hostname, sample_percent, num_epochs, ae_or_clas, my_type, logger)
        i+=1

    fancy_logging("\n\n\n\n\n\n\n\n\n\n\nTRAINING ALL PARENTS DONE.............." + str(time() - start_time), my_hostname, logger)
    fancy_logging("\n\n\n\n\n\n\n\n\n\n\nTRAINING CHILDREN..............", my_hostname, logger)
    start_time = time()
    my_type = "ANCHOR"
    # **********************TRAIN THE DEPENDENTS
    i = 0
    sample_percent = child_sp
    my_quads = node_qhash_map[my_hostname]
    remainder_quads = [item for item in my_quads if item not in my_centroids]
    for quad_to_train in remainder_quads:
        # THE CENTROID FOR THIS PARTICULAR QUADHASH
        parent_quad = get_parent_quad(quad_to_train)
        execute_ml(quad_to_train, parent_quad, my_hostname, sample_percent, num_epochs, ae_or_clas, my_type, logger)
        i += 1

    fancy_logging("TRAINING ALL CHILDREN DONE.............." + str(time() - start_time), my_hostname, logger)