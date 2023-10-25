import os
import torch
import numpy as np
import collections
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm

from threshold import get_mask
from skimage.color import rgb2gray

import xml.etree.ElementTree as ET


class RGBxThermalTreeCrownDataset(Dataset):
    """ Data loader for RGBxThermal images, in pascal VOC format """

    def __init__(self, root, split="train", masks_dir='masks'):
        self.split = split

        # define directories
        # self.rgb_dir = f"others/shadow_removal/shadowformer_imgs"
        # # self.rgb_dir = f"others/brighten_predictions_imgs"
        # self.rgb_dir = f"D:\MetaFusion-main\RESULT_IMAGE_PATH"
        # self.rgb_dir = f"D:\\UMF-CMGR-main\\results\\fused"
        # self.rgb_dir = f"D:\\MFEIF-main\\result\\affine"
        self.rgb_dir = f"{root}/{split}/rgb" # NOTE: remove /all (added for cynthia)
        self.thm_dir = f"{root}/{split}/thermal"
        lbl_dir = 'pseudo_annotations' if split == 'train' else 'gt_annotations' # TODO: 2 => pseudo labels, without => gt
        self.lbl_dir = f"{root}/{split}/{lbl_dir}"
        self.mask_dir = f"{root}/{split}/{masks_dir}"
        self.mask_weights_dir = f'{root}/{split}/masks_weighted_boxes'

        # get filenames (without prefix)
        self.img_names = os.listdir(self.rgb_dir)
        self.thm_img_names = os.listdir(self.thm_dir)
        # self.label_names = os.listdir(self.lbl_dir)
        self.mask_names = os.listdir(self.mask_dir)

        # NOTE: for removing tree less images
        # if split =='train':
        #     img_prefixes = set([name.split('.')[0]  for name in self.img_names])
        #     mask_weights_names = set([name.split('.')[0] for name in os.listdir(self.mask_weights_dir)])
        #     to_remove = img_prefixes - mask_weights_names
        #     for name in to_remove:
        #         self.img_names.remove(f"{name}.tif")
        #     self.mask_weights_names = list([f'{name}.xml.pt' for name in mask_weights_names])


            # NOTE: only keep 08_30 images (along with removing tree less above)
            # self.mask_weights_names = [name for idx,name in enumerate(self.mask_weights_names) if '0830' in self.img_names[idx]]
            # self.img_names = [name for idx,name in enumerate(self.img_names) if '0830' in self.img_names[idx]]

        # NOTE: only keep 08_30 images
        # if split == 'train':
        #     self.mask_names = [name for idx,name in enumerate(self.mask_names) if '0830' in self.img_names[idx]]
        #     self.img_names = [name for idx,name in enumerate(self.img_names) if '0830' in self.img_names[idx]]
        # print(len(self.img_names) , len(self.mask_weights_names))
        # assert len(self.img_names) == len(self.mask_names)




        # default transform
        self.rgb_tform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406],  [0.229, 0.224, 0.225]),
            # transforms.Normalize([0.4199, 0.4298, 0.3222], [0.2377, 0.2336, 0.1907]),
        ])
        self.thm_tform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([13.7140], [1.8340]),
        ])


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load images
        rgb_img = io.imread(f'{self.rgb_dir}/{self.img_names[index]}')
        thm_img = io.imread(f'{self.thm_dir}/{self.thm_img_names[index]}') # NOTE: change!

        # get mask (morphologically determined mask, OR based on pseudo boxes)
        # if self.split != 'train':
        mask = torch.load(f'{self.mask_dir}/{self.mask_names[index]}')
        # else:
        #     mask = torch.load(f'{self.mask_weights_dir}/{self.mask_weights_names[index]}') # NOTE: for box masking (weighted or not)

        # apply transforms
        rgb_img = self.rgb_tform(rgb_img)
        thm_img = self.thm_tform(thm_img)
        # mask = self.rgb_tform(mask)

        return rgb_img, thm_img, mask
    

from torchvision.datasets import CocoDetection

    

def compute_mean_and_std():
    dataset = RGBxThermalTreeCrownDataset(root='data', split='test')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    rgb_mean, rgb_std = 0., 0.
    thm_mean, thm_std = 0., 0.

    for sampled_batch in tqdm(dataloader):
        rgb_img, thm_img, bboxes = sampled_batch
        
        batch_samples = rgb_img.size(0) # batch size (the last batch can have smaller size!) 
        rgb_img = rgb_img.view(batch_samples, rgb_img.size(1), -1) # originally (bsz, 3, 500,500)
        rgb_mean += rgb_img.mean(2).sum(0)
        rgb_std += rgb_img.std(2).sum(0)

        thm_img = thm_img.view(batch_samples, -1)
        thm_mean += thm_img.mean(1).sum(0)
        thm_std += thm_img.std(1).sum(0)


    rgb_mean /= len(dataloader.dataset)
    rgb_std /= len(dataloader.dataset)

    thm_mean /= len(dataloader.dataset)
    thm_std /= len(dataloader.dataset)

    print("RGB => ", rgb_mean, rgb_std)
    print("Thermal => ", thm_mean, thm_std)

if __name__ == "__main__":
    compute_mean_and_std()

