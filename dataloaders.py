""" 
Script implementing PyTorch dataloaders for RT-Trees Dataset 
"""

import os
import torch
from skimage import io

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class RGBxThermalTreeCrownDataset(Dataset):
    """ Data loader for RGBxThermal images, in pascal VOC format """

    def __init__(self, root, split="train", masks_dir='masks'):
        self.split = split

        # define directories
        self.rgb_dir = f"{root}/{split}/rgb" #
        self.thm_dir = f"{root}/{split}/thermal"
        lbl_dir = 'pseudo_annotations' if split == 'train' else 'gt_annotations' # TODO: 2 => pseudo labels, without => gt
        self.lbl_dir = f"{root}/{split}/{lbl_dir}"
        self.mask_dir = f"{root}/{split}/{masks_dir}"
        self.mask_weights_dir = f'{root}/{split}/masks_weighted_boxes'

        # get filenames (without prefix)
        self.img_names = os.listdir(self.rgb_dir)
        self.thm_img_names = os.listdir(self.thm_dir)
        self.label_names = os.listdir(self.lbl_dir)
        self.mask_names = os.listdir(self.mask_dir)


        # default transform (just convert to Tensor) -- normalization done in RetinaNet itself, so not needed here
        self.rgb_tform = transforms.Compose([transforms.ToTensor(),])
        self.thm_tform = transforms.Compose([transforms.ToTensor(),])


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load images
        rgb_img = io.imread(f'{self.rgb_dir}/{self.img_names[index]}')
        thm_img = io.imread(f'{self.thm_dir}/{self.thm_img_names[index]}') # NOTE: change!

        # get mask (morphologically determined mask, OR based on pseudo boxes)
        mask = torch.load(f'{self.mask_dir}/{self.mask_names[index]}')

        # apply transforms
        rgb_img = self.rgb_tform(rgb_img)
        thm_img = self.thm_tform(thm_img)

        return rgb_img, thm_img, mask
    

def compute_mean_and_std():
    """ Compute mean and std of RT-Trees Dataset for normalization statistics """
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

