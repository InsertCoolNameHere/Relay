import math
import os
import torch
import numpy as np
from utilz.im_manipulation import tensor2im, eval_psnr_and_ssim, eval_psnr_and_ssim_old
from model.AutoEnc import RedNet
from data.ImageDataset import ImageDataset
import pytorch_lightning as pl
from collections import OrderedDict, defaultdict
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import socket
import logging
from time import time

n_cpu = 8
file_separator = "/"


# MODULE THAT HOUSES THE GENERATOR AND DISCRIMINATOR
class AETrainer(pl.LightningModule):

    # INIT OF GENERATOR AND DISCRIMINATOR
    def __init__(self, opt=None, base_dir="/nopath", save_dir="data/checkpoints", num_nodes=1, current_scale=0, isret = False):
        super(AETrainer, self).__init__()

        self.input = torch.zeros(opt.train.batch_size, 3, 48, 48, dtype=torch.float32)
        self.true_hr = torch.zeros_like(self.input, dtype=torch.float32)
        self.base_dir = base_dir

        tile_dir = opt.xtra.img_tile_path
        hostname = str(socket.gethostname())
        self.ip_dir = tile_dir.replace("HOSTNAME", hostname, 1)
        self.albums = opt.xtra.albums

        self.opt = opt
        self.save_dir = save_dir
        std = np.asarray(self.opt.train.dataset.stddev)
        mean = np.asarray(self.opt.train.dataset.mean)
        self.denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

        # Dictionary of scalewise best evaluated scores
        self.best_eval = OrderedDict([('psnr_x%d' % s, 0.0) for s in opt.data.scale])
        # Dictionary of scalewise all evaluated scores
        self.eval_dict = OrderedDict([('psnr_x%d' % s, []) for s in opt.data.scale])

        # TENSOR TO NUMPY ARRAY
        self.tensor2im = lambda t: tensor2im(t, mean=opt.train.dataset.mean, stddev=opt.train.dataset.stddev)

        # INITIALIZING THE GENERATOR
        self.model = RedNet()

        self.lr = self.opt.train.lr

        self.l1_criterion = torch.nn.MSELoss()
        self.errors_accum = defaultdict(list)

        training_dataset = ImageDataset(self.albums, None, self.ip_dir, self.opt.xtra.img_type, self.opt.train.dataset.mean,
                                        self.opt.train.dataset.stddev, self.opt.xtra.inp_img_res,
                                        1)
        self.val_dataset = ImageDataset(self.albums, None, self.ip_dir, self.opt.xtra.img_type, self.opt.train.dataset.mean,
                                        self.opt.train.dataset.stddev, self.opt.xtra.inp_img_res,
                                        0.1)


        self.train_dataset = training_dataset

        to_ignore = []
        to_ignore.extend(training_dataset.image_fnames)
        to_ignore.extend(self.opt.ignorables)
        print("RIKI: IGNORED INVALID FILES: ", len(self.opt.ignorables))
        # LOADING TEST_DATA************************************
        self.testing_dataset = ImageDataset(self.albums, None, self.ip_dir, self.opt.xtra.img_type, self.opt.train.dataset.mean,
                                        self.opt.train.dataset.stddev, self.opt.xtra.inp_img_res,
                                        0.1)
        # PERFORM EVALUATION

        self.testing_data_loader = DataLoader(
            self.testing_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=n_cpu,
        )
        # END- LOADING TEST DATA********************************
        print("RIKI: TEST DATA HAS", len(self.testing_data_loader))

        self.psnrs = 0.0
        self.losses = 0.0
        self.losses_l1 = 0.0
        self.psnr_count = 0.0

        self.train_losses = 0.0
        self.train_cnt = 0.0
        self.train_l1 = 0.0
        logging.info("RIKI: BEGINNING EPOCH: 0 "+str(time()))
        self.start_time = time()

    def configure_optimizers(self):
        self.optimizer_G = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.opt.train.lr
            )


        self.scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G,mode='min',factor=0.1,patience=self.opt.train.lr_schedule_patience,
                                                                      verbose=True,min_lr=self.opt.train.D_smallest_lr)

        optimizers = [self.optimizer_G]

        schedulers = [{
                 'scheduler': self.scheduler_G,
                 'monitor': 'val_loss',
                 'interval': 'epoch',
                 'frequency': 1,
                 'strict': True,
                }]

        return optimizers, schedulers

    def val_dataloader(self):
        # LOADING VALIDATION DATA************************************
        self.val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=n_cpu,
        )
        # END- LOADING VALIDATION DATA********************************

        print("RIKI: VAL DATA HAS", len(self.val_data_loader))
        return self.val_data_loader

    def validation_step(self, batch, batch_nb):
        imgs = batch
        self.set_input(imgs['hr'], imgs['hr'])

        self.calc_output()
        self.forward()
        self.compute_loss()

        self.losses_l1 += self.l1_loss
        self.psnr_count += 1

        #self.log("val_loss", self.loss_G, on_epoch=True, logger=True)
        self.log("val_loss", self.loss_G, on_epoch=True, logger=True)

    def train_dataloader(self):

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.opt.train.batch_size,
            shuffle=True,
            num_workers=n_cpu
        )
        # END- LOADING TEST DATA********************************

        print("RIKI: TRAIN DATA HAS", len(dataloader))
        return dataloader

    # THE LR AND THE BICUBIC INTERPOLATED HR IMAGE AS INPUT
    # RETURNS THE GENERATED HR OUTPUT
    def forward(self):
        self.calc_output()
        return self.output

    def calc_output(self):
        self.output = self.model(self.input)

    def set_input(self, lr, hr):
        # print("CURRENT DEVICE...",self.device)
        self.input.resize_(lr.size()).copy_(lr)
        self.true_hr.resize_(hr.size()).copy_(hr).to(self.device)

        self.input = self.input.to(self.device)
        self.true_hr = self.true_hr.to(self.device)

    # GENERATOR LOSS
    def compute_loss(self):
        self.loss_G = 0
        self.l1_loss = torch.sqrt((self.output - self.true_hr).pow(2).mean())
        #self.l1_loss = self.l1_criterion(self.output, self.true_hr)

        self.loss_G += (self.l1_loss)

    # AT THE END, ADJUST LEARNING RATE IF NECESSARY...LATER
    def on_epoch_end(self):
        time_taken = time() - self.start_time

        logging.info('********************RIKI: EPOCH %d TIME TAKEN: %f NOW: %f' % (self.current_epoch, time_taken, time()))
        curr_prog = self.current_epoch + 1

        self.progress = curr_prog

        self.losses_l1 = 0.0
        self.psnr_count = 0.0

        self.start_time = time()

    def on_validation_epoch_end(self) -> None:
        if self.psnr_count > 0:
            avg_l1 = self.losses_l1/self.psnr_count
        else:
            avg_l1 = 0

        logging.info('RIKI: EPOCH %d VALIDATION LOSSES: %s' % (self.current_epoch, str(float(avg_l1))))


    def on_epoch_start(self):
        self.start_time = time()


    def training_step(self, batch, batch_nb):

        imgs = batch

        # logging.info("RIKI: FILENAME IS "+str(imgs['fname']))

        # print("IMAGE LR SHAPE",imgs['lr'].size())
        self.set_input(imgs['hr'], imgs['hr'])

        self.forward()
        self.compute_loss()

        if batch_nb % 100 == 0:

            self.log('train_loss', self.loss_G, on_step=True, on_epoch=True, logger=True)
            logging.info('RIKI:>>>>>> EPOCH %d TRAINING LOSSES: %f' % (self.current_epoch, self.loss_G))


        return {'loss': self.loss_G}

    def get_current_errors(self):
        d = OrderedDict()
        if hasattr(self, 'l1_loss'):
            d['l1_x%d' % self.model_scale] = self.l1_loss.item()

        return d

    def save(self, epoch, lr, scale, make_save=False):
        #(opt_g, opt_d) = self.optimizers()
        to_save = {
            'network': self.save_network(self.model, 'G', str(epoch), str(scale), make_save),
            #'optim': self.save_optimizer(self.optimizer_G, 'G', epoch, lr, str(scale), make_save),
        }

        print("RIKI: SAVED LATEST MODEL %d %d %d...DISK: %s" % (epoch, lr, scale, str(make_save)))
        return to_save

    def save_network(self, network, network_label, epoch_label, scale, make_save=False):
        network = network.module if isinstance(
            network, torch.nn.DataParallel) else network
        # save_filename = '%s_net_%s_x%s.pth' % (epoch_label, network_label, scale)
        save_filename = 'net_%s_x%s.pth' % (network_label, scale)
        save_path = os.path.join(self.save_dir, save_filename)
        to_save = {
            'state_dict': network.state_dict(),
            'path': save_path
        }

        if make_save:
            torch.save(to_save, save_path)

        return to_save

    def save_optimizer(self, optimizer, network_label, epoch, lr, scale, make_save=False):
        save_filename = 'optim_%s_x%s.pth' % (network_label, scale)
        save_path = os.path.join(self.save_dir, save_filename)

        to_save = {
            'state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'lr': lr,
            'path': save_path
        }

        if make_save:
            torch.save(to_save, save_path)

        return to_save

    def load(self, resume_from):
        print("LOADING: ", resume_from[0])
        self.load_network(self.model, resume_from[0])
        # self.load_optimizer(self.optimizer_G, resume_from[1])

    def load_network(self, network, saved_path):
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
        print('RIKI: loaded network state from ' + saved_path)

    def load_optimizer(self, optimizer, saved_path):

        data = torch.load(saved_path)
        loaded_state = data['state_dict']
        optimizer.load_state_dict(loaded_state)

        # Load more params
        self.start_epoch = data['epoch']
        self.lr = data['lr']

        print('RIKI: loaded optimizer state from ' + saved_path)

    def set_eval(self):
        self.model.eval()
        self.isTrain = False

    def set_train(self):
        self.model.train()
        self.model.requires_grad_(True)
        self.discriminator.requires_grad_(True)
        self.isTrain = True

    def reset_eval_result(self):
        for k in self.eval_dict:
            self.eval_dict[k].clear()

    # AVERAGING CURRENT PSNRs FROM eval_dict
    def get_current_eval_result(self):
        eval_result = OrderedDict()
        for k, vs in self.eval_dict.items():
            eval_result[k] = 0
            if vs:
                for v in vs:
                    eval_result[k] += v
                eval_result[k] /= len(vs)
        return eval_result

    def get_current_eval_result_pyr(self):

        return self.eval_dict.copy()

    def update_best_eval_result(self, epoch, current_eval_result=None):
        if current_eval_result is None:
            eval_result = self.get_current_eval_result()
        else:
            eval_result = current_eval_result
        is_best_sofar = any(
            [np.round(eval_result[k], 2) > np.round(v, 2) for k, v in self.best_eval.items()])
        # print("RIKI: trainer IS BEST SO FAR: "+str(is_best_sofar))
        if is_best_sofar:
            self.best_epoch = epoch
            self.best_eval = {
                k: max(self.best_eval[k], eval_result[k])
                for k in self.best_eval
            }

