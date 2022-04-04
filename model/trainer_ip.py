import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image
from torchvision import transforms
from data.ImageDataset import ImageDataset
from model.AutoEnc import Inpainter
from model.Inpaint_model import GatedGenerator
import socket
import utilz.im_manipulation as ImageManipulator
import numpy as np
import os.path as osp
import os
import shutil
import math
from utilz.helper import fancy_logging
import random
from time import time

print_freq = 20
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#device = torch.device('cpu')
mean = [0.4488, 0.4371, 0.404]
stddev = [0.0039215, 0.0039215, 0.0039215]

save_on = False
std = np.asarray(stddev)
mn = np.asarray(mean)
denormalize = transforms.Normalize((-1 * mn / std), (1.0 / std))
l1_criterion = torch.nn.L1Loss()

# SAVING TO MEMORY
def save(network, quadhash, save_dir, make_save=False):

    to_save = {
        'network': save_network(network, quadhash, save_dir, make_save),
    }

    print("RIKI: SAVED LATEST MODEL %d...DISK: %s"%(quadhash, str(make_save)))
    return to_save

def save_network(network, model_id, save_dir, make_save=False):
    network = network.module if isinstance(
        network, torch.nn.DataParallel) else network
    #save_filename = '%s_net_%s_x%s.pth' % (epoch_label, network_label, scale)
    save_filename = 'model_%s.pth' % (str(model_id))
    save_path = os.path.join(save_dir, save_filename)
    to_save = {
        'state_dict': network.state_dict(),
        'path': save_path
    }

    if make_save:
        torch.save(to_save, save_path)

    return to_save

# resume_path is a directory
def load(resume_path, quadhash, model):

    save_path = os.path.join(resume_path, "model_"+quadhash+".pth")
    if os.path.exists(save_path):
        load_network(model, save_path)
        print("LOADED MODEL FROM ", (save_path))

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
    print('RIKI: loaded network state from ' + saved_path)

def train(epoch, train_loader, model, optimizer, base_path):
    # Ensure dropout layers are in train mode
    model.train()

    start = time()

    # Batches
    b_n = 0
    for i_batch, imgs in enumerate(train_loader):
        # Set device options
        x = imgs['composite']
        x1 = imgs['mask']
        y = imgs['hr']
        x = x.to(device)
        x1 = x1.to(device)
        y = y.to(device)
        inv_mask = 1 - imgs['mask']
        inv_mask = inv_mask.to(device)
        # Zero gradients
        optimizer.zero_grad()

        _,y_hat = model(x, x1)

        ghr = denormalize(y_hat[0]*inv_mask[0]).cpu()
        hr = denormalize(y[0]*inv_mask[0]).cpu()
        #print(hr.unsqueeze(0).size(), ghr.unsqueeze(0).size())
        op = ImageManipulator.combine_img_list((hr.unsqueeze(0), ghr.unsqueeze(0)), 2)
        fname =  base_path + str(epoch) + "_" + str(b_n) + ".jpg"
        # print('RIKI: SAVE TO ' + fname)
        if save_on and i_batch == 0:
            print('RIKI: SAVE TO ' + fname)
            save_image(op, fname, normalize=False)

        loss = l1_criterion(y_hat * inv_mask, y * inv_mask)

        #loss = torch.sqrt((y_hat - y).pow(2).mean())
        loss.backward()
        optimizer.step()
        batch_time = time() - start
        b_n+=1
        start = time()

        # Print status
        if i_batch % print_freq == 0:
            print('Epoch: %s Batch %d Batch_time_avg %s Loss %f'%(str(epoch), (i_batch), str(batch_time), loss.item()))


def valid(val_loader, model, tensor2im = None):
    model.eval()  # eval mode (no dropout or batchnorm)
    start = time()

    tot_loss = 0
    i = 0
    model.eval()
    with torch.no_grad():
        # Batches
        psnrs = []
        cc = 0
        for i_batch, imgs in enumerate(val_loader):
            start = time()
            x = imgs['composite']
            x1 = imgs['mask']
            y = imgs['hr']
            x = x.to(device)
            x1 = x1.to(device)
            y = y.to(device)

            _,y_hat = model(x,x1)

            #loss1 = torch.sqrt((y_hat - y).pow(2).mean())
            inv_mask = 1 - imgs['mask']
            inv_mask = inv_mask.to(device)

            #print(y_hat.get_device(), y.get_device(), inv_mask.get_device())

            loss1 = l1_criterion(y_hat * inv_mask, y * inv_mask)

            im1 = tensor2im(y* inv_mask)
            im2 = tensor2im(y_hat* inv_mask)
            psnrval = ImageManipulator.eval_psnr_and_ssim(im1, im2)[0]
            if math.isfinite(psnrval):
                psnrval = math.floor(psnrval*100)/100
                cc+=1
            else:
                psnrval = 99999
            psnrs.append(psnrval)

            # Keep track of metrics
            batch_time = time() - start
            #tot_loss+=psnrval
            tot_loss += loss1.item()
            i +=1

    psnrs.sort()
    if len(psnrs) < 3:
        #print("HERE>>>>>>>>>>>>>>>>>>>")
        avg_psnr = sum(psnrs)/cc
    else:
        leng = len(psnrs)
        n = int(max(leng/10,1))
        psnrs = psnrs[n:-n]
        avg_psnr = sum(psnrs) / (cc - 2*n)

    #val_loss = 1/((tot_loss/i))
    val_loss = tot_loss / i
    print("BATCH_TIME: ", batch_time, "PSNR: ", my_rounder(avg_psnr), " VAL LOSS:", my_rounder(val_loss))
    return val_loss


def my_rounder(fl_val):
    fl_val = math.floor(fl_val*100)
    return str(fl_val/100)

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

def adjust_learning_rate(optimizer, shrink_factor, least_lr):
    for param_group in optimizer.param_groups:
        if param_group['lr'] > least_lr:
            param_group['lr'] = param_group['lr'] * shrink_factor
            print("\nDECAYING learning rate.")
            print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
        else:
            print("MINIMUM LR ACHIEVED...NO MORE SHRINK!!!")

# CREATE A FROM-SRATCH MODEL
def get_base_model_IP():
    model = GatedGenerator()
    return model

def prep_model(model, transfer_learning):
    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
        model.re_init_conv()
        model.re_init_final_deconv()
        model = model.to(device)
    else:
        # Use appropriate device
        model = model.to(device)
    return model

def modeling_ip(quadhash, parent_err, transferLearning, model, hostname, epochs, sample_percent, logger = None):
    # CONSTANTS
    random.seed(7)
    batch_size = 8
    albums = ['co-3month']
    img_dir = "/s/" + hostname + "/a/nobackup/galileo/stip-images/co-3month/Sentinel-2/"
    save_image_path = "/s/" + hostname + "/a/nobackup/spectral/sapmitra/SRImages/saved_ae/"
    clear_or_create(save_image_path)
    img_type = '-3.tif'
    input_image_res = 64
    # NUMBER OF ALLOWABLE EPOCHS OF NO IMPROVEMENT

    # PREPPING THE MODEL BEFORE TRAINING
    model = prep_model(model, transferLearning)

    tensor2img = lambda t: ImageManipulator.tensor2im(t, mean=mean, stddev=stddev)

    start_lr = 1e-4
    least_lr = 0.000001
    shrinkage = 0.1
    # NUMBER OF ALLOWABLE EPOCHS OF NO IMPROVEMENT
    upd_patience = 8
    patience = 20

    optimizer = optim.Adam(model.parameters(), lr=start_lr, betas=(0.9, 0.999))

    training_dataset = ImageDataset(albums, quadhash, img_dir, img_type, mean, stddev, input_image_res, sample_percent)
    val_dataset = ImageDataset(albums, quadhash, img_dir, img_type, mean, stddev, input_image_res, sample_percent * 0.2)

    print(quadhash,":TOTAL TRAINING FILES : ", len(training_dataset.image_fnames))
    print(quadhash,":TOTAL VALIDATION FILES : ", len(val_dataset.image_fnames))

    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    best_loss = 100000
    epochs_since_improvement = 0

    start_time = time()

    saved_state_dict = model.state_dict()

    # Epochs
    i = 0
    for epoch in range(0, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == patience:
            print("LEARNING STAGNATED.....STOPPING TRAINING")
            break

        if epochs_since_improvement % upd_patience == 0 and epochs_since_improvement > 1:
            adjust_learning_rate(optimizer, shrinkage, least_lr)
        # TRAIN STEP
        train(epoch, train_loader, model, optimizer, save_image_path)
        # VALIDATION STEP
        val_loss = valid(val_loader, model, tensor2img)

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        sd = model.state_dict()

        # IN CASE OF PARENT, THE ERROR IS 0, SO NEVER SATISFIED
        if best_loss <= 1.05*parent_err:
            saved_state_dict = sd
            print("TARGET ERROR ACHIEVED....EARLY STOPPING")
            break

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d" % (epochs_since_improvement,))
        else:
            saved_state_dict = sd
            epochs_since_improvement = 0
        i += 1

    best_model = get_base_model_IP()
    best_model.load_state_dict(saved_state_dict)
    print("TRAINING COMPLETE FOR %s" % quadhash, " TIME TAKEN:", str(time() - start_time))
    #fancy_logging("TRAINING COMPLETE FOR %s TRANSFER LEARNING? %s TRAINING_TIME1: %s"%(quadhash, str(transferLearning), str(time() - start_time)), hostname, logger)

    return best_model, best_loss, i

def adjust_learning_rateX(lr_in, optimizer, epoch, lr_decrease_factor, lr_decrease_epoch):
    """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
    lr = lr_in * (lr_decrease_factor ** (epoch // lr_decrease_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    random.seed(7)
    batch_size = 8
    epochs = 10
    hostname = str(socket.gethostname())
    albums = ['co-3month']
    #quadhash = "02132223300"
    quadhash = "02132221313"
    img_dir = "/s/"+hostname+"/a/nobackup/galileo/stip-images/co-3month/Sentinel-2/"
    base_path = "/s/" + hostname + "/a/nobackup/galileo/sapmitra/SRImages/saved_ae/"
    clear_or_create(base_path)
    img_type = '-3.tif'
    input_image_res = 64
    sample_percent = 1
    upd_patience = 8
    patience = 20

    training_dataset = ImageDataset(albums, quadhash, img_dir, img_type, mean, stddev, input_image_res, sample_percent)
    val_dataset = ImageDataset(albums, quadhash, img_dir, img_type, mean, stddev, input_image_res, 0.2)

    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    tensor2img = lambda t: ImageManipulator.tensor2im(t, mean=mean, stddev=stddev)


    parent_start_time = time()
    #model = get_base_model_AE()
    model = get_base_model_IP()

    # Use appropriate device
    model = model.to(device)
    # print(model)
    start_lr = 1e-4
    least_lr = 0.000001
    shrinkage = 0.1
    #optimizer = optim.Adam(model.parameters(), lr=start_lr)
    optimizer = optim.Adam(model.parameters(), lr=start_lr, betas=(0.9, 0.999))

    best_loss = 100000
    epochs_since_improvement = 0

    # Epochs
    i = 0
    for epoch in range(0, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == patience:
            print("LEARNING STAGNATED.....STOPPING TRAINING")
            break

        if epochs_since_improvement % upd_patience == 0 and epochs_since_improvement>1:
            adjust_learning_rate(optimizer, shrinkage, least_lr)

        # One epoch's training
        train(epoch, train_loader, model, optimizer, base_path)

        # One epoch's validation
        val_loss = valid(val_loader, model, tensor2img)

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d BEST LOSS: %f" % (epochs_since_improvement,best_loss))
        else:
            epochs_since_improvement = 0
        i+=1

        #adjust_learning_rate(start_lr, optimizer, epoch + 1, 0.5, 10)

    parent_epochs = i
    parent_loss = best_loss
    parent_training_interval = time() - parent_start_time

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Save checkpoint
    #save_checkpoint(i, model, optimizer, val_loss, is_best)

    # TRANSFER LEARNING ******************************************************************
    print("TRANSFER LEARNING ******************************************************************")
    random.seed(7)
    clear_or_create(base_path)

    quadhash = "02132221313"
    training_dataset = ImageDataset(albums, quadhash, img_dir, img_type, mean, stddev, input_image_res, sample_percent)
    val_dataset = ImageDataset(albums, quadhash, img_dir, img_type, mean, stddev, input_image_res, 0.2)

    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    child_start_time = time()


    for param in model.parameters():
        param.requires_grad = False
    model.re_init_conv()
    model.re_init_final_deconv()

    model = model.to(device)
    start_lr = 1e-4
    least_lr = 0.000001
    shrinkage = 0.5

    optimizer = optim.Adam(model.parameters(), lr=start_lr, betas=(0.5, 0.999))
    best_loss = 100000
    epochs_since_improvement = 0
    # Epochs
    i = 0
    for epoch in range(0, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == patience:
            print("LEARNING STAGNATED.....STOPPING TRAINING")
            break
        if best_loss <= 1.05 * parent_loss:
            print("TARGET ACHIEVED.....STOPPING TRAINING")
            break

        if epochs_since_improvement % upd_patience == 0 and epochs_since_improvement>1:
            adjust_learning_rate(optimizer, shrinkage, least_lr)

        # One epoch's training
        train(epoch, train_loader, model, optimizer, base_path)

        # One epoch's validation
        val_loss = valid(val_loader, model, tensor2img)

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d BEST LOSS: %f" % (epochs_since_improvement,best_loss))
        else:
            epochs_since_improvement = 0
        i += 1

    child_training_interval = time() - child_start_time

    print("\n\nSUMMARY: PARENT TIME: %s (%d) PARENT LOSS: %s CHILD TIME: %s (%d) CHILD LOSS %s"%(str(parent_training_interval), parent_epochs, str(parent_loss),
                                                                                             str(child_training_interval), i, str(best_loss)))
if __name__ == '__main__':
    main()