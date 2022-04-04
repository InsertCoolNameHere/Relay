from data.NLCDDataset import NLCDDataset
import random
from torch.utils.data import DataLoader
import logging
import os
import os.path as osp
import socket

my_hostname = str(socket.gethostname())
def config_log():
    base_path = "/s/" + my_hostname + "/a/nobackup/spectral/sapmitra/"
    log_file_name = base_path + 'nlcd_' + my_hostname + '.log'

    if not osp.isdir(base_path):
        os.makedirs(base_path)

    clear_old_log(log_file_name)
    logging.basicConfig(format='%(message)s',filename=log_file_name, level=logging.INFO)

def clear_old_log(old_log):
    if os.path.exists(old_log):
        print("REMOVED OLD LOG")
        os.remove(old_log)

if __name__ == '__main__':
    config_log()
    random.seed(7)
    batch_size = 1
    albums = ['conus-usa']
    img_dir = "/s/" + my_hostname + "/a/nobackup/galileo/stip-images/ALBUM/"
    #img_dir = "/s/chopin/e/proj/sustain/sapmitra/NLCD_test/"
    img_type = '.tif'

    training_dataset = NLCDDataset(albums, img_dir, img_type)
    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)

    for i_batch, imgs in enumerate(train_loader):
        if i_batch%100 == 0:
            print(imgs["qhash"], imgs["content"])