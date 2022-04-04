from __future__ import print_function, division
from torch.utils.data import Dataset
import os
import gdal
import numpy as np
import logging

class NLCDDataset(Dataset):

    codes = [0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95]

    # RETURNS A HIGH RES VERSION OF THE IMAGE AT A GIVEN INDEX
    def __getitem__(self, index):
        content_dict = []
        for c in range(len(self.codes)):
            content_dict.append(0.0)

        ret_data = {}
        # Load HIGH-RES image
        if len(self.image_fnames):
            filename = self.image_fnames[index]
            tokens = filename.split('/')
            ln = len(tokens)

            image_hash = tokens[ln - 3]
            ret_data["qhash"] = image_hash
            # FULL HIGH RESOLUTION IMAGE
            target_img = basic_image_loader_gdal(filename)

            #print(target_img)
            (unique, counts) = np.unique(target_img, return_counts=True)
            tot_sum = np.sum(counts)

            for i in range(len(unique)):
                frac = float(counts[i])/float(tot_sum)
                indx = self.codes.index(unique[i])
                content_dict[indx] = frac

            converted_list = [str(element) for element in content_dict]
            retr_str = image_hash+","+",".join(converted_list)
            ret_data["content"] = retr_str
            logging.info(retr_str)
        return ret_data

    def __len__(self):
        return len(self.image_fnames)

    def __init__(self, albums, img_dir, img_type):
        self.img_dir = img_dir

        # COLLECTING ALL TRAINING/TESTING IMAGE PATHS
        all_images = []
        for alb in albums:
            imd = img_dir.replace("ALBUM", alb)
            all_images.extend(get_filenames(imd, img_type))

        self.image_fnames = all_images

# RETURNS ALL FILENAMES IN THE GIVEN DIRECTORY IN AN ARRAY
def get_filenames(source, ext):
    img_paths = []
    for roots, dir, files in os.walk(source):
        for file in files:
            if file.endswith(ext):
                file_abs_path = os.path.join(roots, file)
                img_paths.append(file_abs_path)
    return img_paths

def basic_image_loader_gdal(filename):
    gdal_obj = gdal.Open(filename)
    nlcds = gdal_obj.GetRasterBand(1).ReadAsArray()
    return nlcds





