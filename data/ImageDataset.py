from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from random import sample
from PIL import Image
import utilz.im_manipulation as ImageManipulator
import utilz.mask_utils as Mask_Utils

class ImageDataset(Dataset):

    # RETURNS A HIGH RES VERSION OF THE IMAGE AT A GIVEN INDEX
    def __getitem__(self, index):
        ret_data = {}

        # Load HIGH-RES image
        if len(self.image_fnames):

            filename = self.image_fnames[index]
            tokens = filename.split('/')
            ln = len(tokens)
            # print(filename, tokens)
            # QUAD HASH OF THE ENTIRE IMAGE
            image_hash = tokens[ln - 3]

            target_hash = ImageManipulator.get_candidate_tiles(image_hash, 13, self.randomHash)

            # FULL HIGH RESOLUTION IMAGE
            target_img = ImageManipulator.image_loader_gdal(filename, target_hash)
            target_img = ImageManipulator.imageResize(target_img, self.highest_res)

            hr_image = target_img

            ret_data['hr'] = hr_image
            ret_data['hr'] = self.normalize_fn(ret_data['hr'])

            puncture_percent = 0.25
            grid_size = 4
            # FINAL_MASK HAS 0s IN PLACES FOR WHICH HR INFO IS MISSING
            tile_wh = int(self.highest_res/grid_size)
            final_mask, total_masks = Mask_Utils.create_mask_for_puncture(grid_size, puncture_percent, tile_wh)

            final_mask = final_mask.astype('float32')
            final_mask = self.tensorise_fn(final_mask)
            # SUPERIMPOSED VERSION OF HR AND HR'
            ret_data["total_masks"] = total_masks
            ret_data["mask"] = final_mask
            ret_data["inv_mask"] = 1 - final_mask
            ret_data["puncture_percent"] = puncture_percent

            # COMPOSITE SR IMAGE
            #print(ret_data['hr'].size(), final_mask.size())
            ret_data['composite'] = ret_data['hr'] * final_mask
            #print(type(puncture_percent), final_mask.dtype, ret_data['composite'].dtype, ret_data['hr'].dtype)

        return ret_data

    def __len__(self):
        return len(self.image_fnames)


    def __init__(self, albums, quadhash, img_dir, img_type, mean, stddev, input_image_res, sample_percent, fixedHash = False):
        self.mean = mean
        self.stddev = stddev

        self.randomHash = True

        # THE RESOLUTION OF THE x8 ACTUAL IMAGE
        self.highest_res = input_image_res
        # THE DIRECTIRY WHERE THE REAL HR IMAGES ARE
        self.img_dir = img_dir

        # COLLECTING ALL TRAINING/TESTING IMAGE PATHS
        all_images = []
        for alb in albums:
            imd = img_dir.replace("ALBUM", alb)
            if quadhash:
                all_images.extend(ImageManipulator.get_filenames(imd, img_type, quadhash))
            else:
                all_images.extend(ImageManipulator.get_filenames_dist(imd, img_type))

        #print("TOTAL TIFS", len(all_images))

        # SAMPLE OR NOT?
        if sample_percent < 1:
            num_inputs = len(all_images) * sample_percent
            num_inputs = max(num_inputs,1)
            self.image_fnames = sample(all_images, int(num_inputs))
            if fixedHash:
                self.randomHash = False
        else:
            self.image_fnames = all_images

        # Input normalization
        self.normalize_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.stddev)
        ])
        self.tensorise_fn = transforms.Compose([
            transforms.ToTensor()
        ])

        self.normalize_fn_1d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5),
                                 (0.5))
        ])




