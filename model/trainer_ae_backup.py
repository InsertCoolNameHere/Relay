import time

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image
from torchvision import transforms
from data.ImageDataset import ImageDataset
from model.AutoEnc import SegNet
import socket
import utilz.im_manipulation as ImageManipulator
import numpy as np
import os.path as osp
import os
import shutil

print_freq = 10
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#device = torch.device('cpu')
mean = [0.4488, 0.4371, 0.404]
stddev = [0.0039215, 0.0039215, 0.0039215]

std = np.asarray(stddev)
mn = np.asarray(mean)
denormalize = transforms.Normalize((-1 * mn / std), (1.0 / std))

def train(epoch, train_loader, model, optimizer, base_path):
    # Ensure dropout layers are in train mode
    model.train()

    start = time.time()

    # Batches
    b_n = 0
    for i_batch, imgs in enumerate(train_loader):
        # Set device options
        x = imgs['hr']
        y = imgs['hr']
        x = x.to(device)
        y = y.to(device)

        # print('x.size(): ' + str(x.size())) # [32, 3, 224, 224]
        # print('y.size(): ' + str(y.size())) # [32, 3, 224, 224]

        # Zero gradients
        optimizer.zero_grad()

        y_hat = model(x)

        ghr = denormalize(y_hat[0]).cpu()
        hr = denormalize(imgs['hr'][0]).cpu()
        op = ImageManipulator.combine_img_list((hr.unsqueeze(0), ghr.unsqueeze(0)), 2)
        fname =  base_path + str(epoch) + "_" + str(b_n) + ".jpg"
        # print('RIKI: SAVE TO ' + fname)
        if i_batch%3 == 0:
            save_image(op, fname, normalize=False)

        loss = torch.sqrt((y_hat - y).pow(2).mean())
        loss.backward()

        # optimizer.step(closure)
        optimizer.step()
        batch_time = time.time() - start
        b_n+=1
        start = time.time()

        # Print status
        if i_batch % print_freq == 0:
            print('Epoch: %s Batch %d Batch_time_avg %s Loss %f'%(str(epoch), (i_batch), str(batch_time), loss))


def valid(val_loader, model):
    model.eval()  # eval mode (no dropout or batchnorm)
    start = time.time()

    tot_loss = 0
    i = 0
    with torch.no_grad():
        # Batches
        for i_batch, imgs in enumerate(val_loader):
            x = imgs['hr']
            y = imgs['hr']
            # Set device options
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)

            loss = torch.sqrt((y_hat - y).pow(2).mean())

            # Keep track of metrics
            batch_time = time.time() - start

            start = time.time()

            tot_loss+=loss
            i +=1

    val_loss = tot_loss/i
    print("BATCH_TIME: ", batch_time, " VAL LOSS:", val_loss)
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

def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def main():
    batch_size = 8
    epochs = 200
    hostname = str(socket.gethostname())
    albums = ['co-3month']
    quadhash = "02132221320"
    img_dir = "/s/"+hostname+"/a/nobackup/galileo/stip-images/co-3month/Sentinel-2/"
    base_path = "/s/" + hostname + "/a/nobackup/galileo/sapmitra/SRImages/saved_ae/"
    clear_or_create(base_path)
    img_type = '-3.tif'
    input_image_res = 64
    sample_percent = 1
    patience = 50


    training_dataset = ImageDataset(albums, quadhash, img_dir, img_type, mean, stddev, input_image_res, sample_percent)
    val_dataset = ImageDataset(albums, quadhash, img_dir, img_type, mean, stddev, input_image_res, 0.5)

    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    # Create SegNet model
    label_nbr = 3
    model = SegNet(label_nbr)

    # Use appropriate device
    model = model.to(device)
    # print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_loss = 100000
    epochs_since_improvement = 0

    # Epochs
    i = 0
    for epoch in range(0, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == patience:
            break

        #if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            #adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(epoch, train_loader, model, optimizer, base_path)

        # One epoch's validation
        val_loss = valid(val_loader, model)
        #print('\n * LOSS - {loss:.3f}\n'.format(loss=val_loss))

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        i+=1

    # Save checkpoint
    #save_checkpoint(i, model, optimizer, val_loss, is_best)


if __name__ == '__main__':
    main()