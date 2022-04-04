import math
import os
import os.path as osp
import torch
import numpy as np
from utilz.im_manipulation import tensor2im,eval_psnr_and_ssim
from data.ImageDataset import ImageDataset
import utilz.im_manipulation as ImageManipulator
from torchvision.utils import save_image, make_grid
import pytorch_lightning as pl
from collections import OrderedDict,defaultdict
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import socket
import logging
from time import time,sleep
from model.AutoEnc import RedNet
from data.mongo_model_utils import *

n_cpu = 8
file_separator = "/"
file_base='/s/chopin/b/grad/sapmitra/status/'
# MODULE THAT HOUSES THE GENERATOR AND DISCRIMINATOR

class AETrainerLight(pl.LightningModule):

    # INIT OF GENERATOR AND DISCRIMINATOR
    def __init__(self, sample_percent):
        super(AETrainerLight, self).__init__()
        self.save_on = False
        random.seed(7)

        if torch.cuda.is_available():
            self.my_device = torch.device('cuda')
        else:
            self.my_device = torch.device('cpu')

        self.batch_size = 8
        self.hostname = str(socket.gethostname())
        self.albums = ['co-3month']

        self.img_dir = "/s/" + self.hostname + "/a/nobackup/galileo/stip-images/co-3month/Sentinel-2/"
        self.base_path = "/s/" + self.hostname + "/a/nobackup/galileo/sapmitra/SRImages/saved_ae/"
        self.img_type = '-3.tif'
        self.input_image_res = 64
        self.sample_percent = sample_percent
        self.upd_patience = 4
        self.start_lr = 0.0015
        self.least_lr = 0.0001
        self.shrinkage = 0.2


        mean = [0.4488, 0.4371, 0.404]
        stddev = [0.0039215, 0.0039215, 0.0039215]

        stddev = np.asarray(stddev)
        mean = np.asarray(mean)
        self.denormalize = transforms.Normalize((-1 * mean / stddev), (1.0 / stddev))


        training_dataset = ImageDataset(self.albums, None, self.img_dir, self.img_type, mean, stddev, self.input_image_res, self.sample_percent)
        self.train_dataset = training_dataset

        self.testing_dataset = ImageDataset(self.albums, None, self.img_dir, self.img_type, mean, stddev, self.input_image_res, 0.1)
        self.val_dataset = ImageDataset(self.albums, None, self.img_dir, self.img_type, mean, stddev, self.input_image_res, 0.1)

        self.testing_data_loader = DataLoader(
            self.testing_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=n_cpu,
        )
        print("RIKI: TEST DATA HAS", len(self.testing_data_loader))

        self.model = RedNet()
        self.l1_criterion = torch.nn.MSELoss()
        self.errors_accum = defaultdict(list)

        self.set_train()

        self.psnrs = 0
        self.cc = 0

        self.train_losses = 0.0
        self.train_cnt = 0.0

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ENDOER-DECODER^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        self.start_time = time()

    def configure_optimizers(self):

        self.optimizer_1  = torch.optim.Adam(self.model.parameters(), lr=self.start_lr)

        self.scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_1, mode='min', factor=self.shrinkage,
                                                                      patience=self.upd_patience, verbose=True, min_lr=self.least_lr)

        optimizers = [self.optimizer_1]
        schedulers = [{
                'scheduler': self.scheduler_1,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
            }
        ]

        return optimizers, schedulers

    def val_dataloader(self):
        # LOADING VALIDATION DATA************************************
        self.val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=n_cpu,
        )
        # END- LOADING VALIDATION DATA********************************

        print("RIKI: VAL DATA HAS", len(self.val_data_loader))
        return self.val_data_loader

    def validation_step(self, batch, batch_nb):

        #print("BEGIN VALIDATION>>>>>>>>>>>>"+self.hostname)
        imgs = batch

        self.psnrs = 0
        self.cc = 0

        x = imgs['hr']
        y = imgs['hr']
        # Set device options
        x = x.to(self.my_device)
        y = y.to(self.my_device)

        y_hat = self.model(x)

        #loss1 = torch.sqrt((y_hat - y).pow(2).mean())
        im1 = tensor2im(y)
        im2 = tensor2im(y_hat)
        psnrval = ImageManipulator.eval_psnr_and_ssim(im1, im2)[0]

        self.val_loss1 = 99999
        if math.isfinite(psnrval):
            psnrval = math.floor(psnrval * 100) / 100
            self.psnrs += psnrval
            self.cc += 1
            self.val_loss1 = 1.0 / (self.psnrs / self.cc)

        self.log("val_loss", self.val_loss1, on_epoch=True, logger=True)
        #print("END VALIDATION>>>>>>>>>>>>" + self.hostname)

    def on_validation_epoch_end(self) -> None:
        avg_psnr = 1.0
        if self.cc > 0:
            avg_psnr = self.psnrs / self.cc

        #print('RIKI: EPOCH %d PSNRS: %f' % (self.current_epoch, avg_psnr))
        logging.info('RIKI: EPOCH %d PSNRS: %f' % (self.current_epoch, avg_psnr))
        self.psnrs = 0
        self.cc = 0

        if self.current_epoch == 25:
            logging.info('RIKI: AT_EPOCH %d CURRENT_TIME: %f' % (self.current_epoch, time()))
            if "lattice-181" in self.hostname:
                print("*****************SAVING GLOBAL MODEL AT ", self.hostname, "TIME TAKEN: ", (time()-self.start_time))
                save_model(self.model, "GLOBAL", {}, self.val_loss1, 1, isTL=True)
        if self.current_epoch == 25:
            if "lattice-181" in self.hostname:
                print("EPOCH: ", self.current_epoch," :TIME TAKEN: ", (time()-self.start_time))




    def train_dataloader(self) :

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_cpu,
        )
        print("RIKI: TRAIN DATA HAS", len(dataloader))
        return dataloader


    def training_step(self, batch, batch_nb):

        #print("TRAINING BATCH: ", self.current_epoch, batch_nb, self.hostname)
        imgs = batch
        x = imgs['hr']
        y = imgs['hr']
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.model(x)

        ghr = self.denormalize(y_hat[0]).cpu()
        hr = self.denormalize(imgs['hr'][0]).cpu()
        op = ImageManipulator.combine_img_list((hr.unsqueeze(0), ghr.unsqueeze(0)), 2)
        fname = self.base_path + str(self.current_epoch) + "_" + str(batch_nb) + ".jpg"
        # print('RIKI: SAVE TO ' + fname)
        if self.save_on and batch_nb == 0:
            print('RIKI: SAVE TO ' + fname)
            save_image(op, fname, normalize=False)

        loss = torch.sqrt((y_hat - y).pow(2).mean())
        return {'loss': loss}

    def set_train(self):
        self.model.train()
        self.model.requires_grad_(True)
        self.isTrain = True
