import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image
from torchvision import transforms
from data.ImageDataset import ImageDataset
from model.AutoEnc import RedNet,Inpainter
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

    #print(">>>>>>>>>>>>>>>>")
    # Batches
    b_n = 0
    for i_batch, imgs in enumerate(train_loader):
        # Set device options
        x = imgs['hr']
        y = imgs['hr']
        x = x.to(device)
        y = y.to(device)
        # Zero gradients
        optimizer.zero_grad()

        y_hat = model(x)

        ghr = denormalize(y_hat[0]).cpu()
        hr = denormalize(imgs['hr'][0]).cpu()
        op = ImageManipulator.combine_img_list((hr.unsqueeze(0), ghr.unsqueeze(0)), 2)
        fname =  base_path + str(epoch) + "_" + str(b_n) + ".jpg"
        # print('RIKI: SAVE TO ' + fname)
        if save_on and i_batch == 0:
            print('RIKI: SAVE TO ' + fname)
            save_image(op, fname, normalize=False)

        loss = torch.sqrt((y_hat - y).pow(2).mean())
        loss.backward()

        # optimizer.step(closure)
        optimizer.step()
        batch_time = time() - start
        b_n+=1
        start = time()

        # Print status
        '''if i_batch % print_freq == 0:
            print('Epoch: %s Batch %d Batch_time_avg %s Loss %f'%(str(epoch), (i_batch), str(batch_time), loss.item()))'''


def valid(val_loader, model, tensor2im = None):
    model.eval()  # eval mode (no dropout or batchnorm)
    start = time()

    tot_loss = 0
    i = 0
    model.eval()
    with torch.no_grad():
        # Batches
        psnrs = 0
        cc = 0
        for i_batch, imgs in enumerate(val_loader):
            x = imgs['hr']
            y = imgs['hr']
            # Set device options
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)

            loss1 = torch.sqrt((y_hat - y).pow(2).mean())
            im1 = tensor2im(y)
            im2 = tensor2im(y_hat)
            psnrval = ImageManipulator.eval_psnr_and_ssim(im1, im2)[0]
            if math.isfinite(psnrval):
                psnrval = math.floor(psnrval*100)/100
                psnrs+=psnrval
                cc+=1
                loss = 1/psnrval
            else:
                loss = 99999

            # Keep track of metrics
            batch_time = time() - start

            start = time()

            tot_loss+=loss
            #tot_loss += loss1.item()
            i +=1

    val_loss = 1.0/(psnrs/cc)
    print("BATCH_TIME: ", batch_time, "PSNR: ", (psnrs/cc), " VAL LOSS:", val_loss)
    return val_loss

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
def get_base_model_AE():
    model = RedNet()
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

def modeling_ae(quadhash, parent_err, transferLearning, model, hostname, epochs, sample_percent, logger = None):
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
    upd_patience = 4
    patience = 24

    # PREPPING THE MODEL BEFORE TRAINING
    model = prep_model(model, transferLearning)

    training_dataset = ImageDataset(albums, quadhash, img_dir, img_type, mean, stddev, input_image_res, sample_percent)
    val_dataset = ImageDataset(albums, quadhash, img_dir, img_type, mean, stddev, input_image_res, sample_percent*0.2)

    print(quadhash, ":TOTAL TRAINING FILES : ", len(training_dataset.image_fnames))
    print(quadhash, ":TOTAL VALIDATION FILES : ", len(val_dataset.image_fnames))

    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    tensor2img = lambda t: ImageManipulator.tensor2im(t, mean=mean, stddev=stddev)

    start_lr = 0.0015
    least_lr = 0.0001
    shrinkage = 0.2
    optimizer = optim.Adam(model.parameters(), lr=start_lr)

    best_loss = 100000
    epochs_since_improvement = 0

    start_time = time()
    # Epochs
    i = 0
    for epoch in range(0, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == patience:
            print("LEARNING STAGNATED.....STOPPING TRAINING")
            break

        if epochs_since_improvement % upd_patience == 0 and epochs_since_improvement>1:
            adjust_learning_rate(optimizer, shrinkage, least_lr)

        # TRAIN STEP
        train(epoch, train_loader, model, optimizer, save_image_path)
        # VALIDATION STEP
        val_loss = valid(val_loader, model, tensor2img)

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if best_loss <= 1.05*parent_err:
            print("TARGET ERROR ACHIEVED....EARLY STOPPING")
            break

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        i += 1
    print("TRAINING COMPLETE FOR %s"%quadhash, " TIME TAKEN:", str(time() - start_time))
    #fancy_logging("TRAINING COMPLETE FOR %s TRANSFER LEARNING? %s TRAINING_TIME1: %s"%(quadhash, str(transferLearning), str(time() - start_time)), hostname, logger)

    return model, best_loss, i

def main():
    random.seed(7)
    batch_size = 8
    epochs = 200
    hostname = str(socket.gethostname())
    albums = ['co-3month']
    quadhash = "02132223300"
    #quadhash = "02132221313"
    img_dir = "/s/"+hostname+"/a/nobackup/galileo/stip-images/co-3month/Sentinel-2/"
    base_path = "/s/" + hostname + "/a/nobackup/galileo/sapmitra/SRImages/saved_ae/"
    clear_or_create(base_path)
    img_type = '-3.tif'
    input_image_res = 64
    sample_percent = 1
    upd_patience = 4
    patience = 24

    training_dataset = ImageDataset(albums, quadhash, img_dir, img_type, mean, stddev, input_image_res, sample_percent)
    val_dataset = ImageDataset(albums, quadhash, img_dir, img_type, mean, stddev, input_image_res, 0.2)

    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    tensor2img = lambda t: ImageManipulator.tensor2im(t, mean=mean, stddev=stddev)


    parent_start_time = time()
    model = get_base_model_AE()

    # Use appropriate device
    model = model.to(device)
    # print(model)
    start_lr = 0.0015
    # start_lr = 0.0025
    least_lr = 0.0001
    shrinkage = 0.2
    optimizer = optim.Adam(model.parameters(), lr=start_lr)

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
            print("\nEpochs since last improvement: %d BEST LOSS: %f" % (epochs_since_improvement,1.0/best_loss))
        else:
            epochs_since_improvement = 0
        i+=1
    parent_epochs = i
    parent_loss = best_loss
    parent_training_interval = time() - parent_start_time
    if True:
        return
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
    start_lr = 0.0015
    least_lr = 0.00033
    shrinkage = 0.66
    optimizer = optim.Adam(model.parameters(), lr=start_lr)

    best_loss = 100000
    epochs_since_improvement = 0
    epochs = 200
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