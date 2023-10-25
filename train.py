"""
Script to train the proposed method
"""

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from dataloaders import RGBxThermalTreeCrownDataset
from archs.RGBTdetector import RGBxThermalDetector
from df_repo.deepforest import main as df_main

 
def train_model(save_dir):
    # data loaders
    train_dataset = RGBxThermalTreeCrownDataset(root='data', split='train', masks_dir='masks')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    # for logging
    writer = SummaryWriter(save_dir)
    checkpoint_dir = f'{save_dir}/checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # define dual-branch proposed model, initialized with Deepforest weights
    weights_path = 'df_retinanet.pt'
    model = RGBxThermalDetector(weights_path)
    model.train()
    model.cuda()
    model.rgb_detector.eval()

    print("Layers to be trained:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # define optimizer and scheduler 
    optimizer = torch.optim.Adam(params = model.parameters())
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    # variables to keep track of during training
    n = 1
    start = 0
    current_epoch = 0
    max_iter = 10000 
    save_step =  1000 
    data_len = len(train_dataloader)
    max_epoch = max_iter / data_len
    loader_iter = iter(train_dataloader)
    loss = None

    # loop until max iters reached
    for i in (pbar:=tqdm(range(start,max_iter))):
        pbar.set_description(f"{loss}")

        # get current batch (reset between epochs)
        try:
            current_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_dataloader)
            current_batch = next(loader_iter)
            current_epoch += 1
            n = 1

            # step scheduler after each epoch
            scheduler.step()
            
        # get inputs on GPU
        rgb_img, thm_img, masks = current_batch
        rgb_img, thm_img, masks = rgb_img.cuda(), thm_img.cuda(), masks.cuda()
        n+=1

        # compute alpha according to GDL paper
        p = float( n + current_epoch * data_len) / max_epoch / data_len
        alpha = 2. / ( 1. + np.exp( -10 * p)) - 1
        alphas = [alpha, alpha, alpha]

        # log losses
        losses_dict = model(rgb_img, thm_img, alphas, masks, comp_fpn_loss = True)
        for key,loss in losses_dict.items():
            writer.add_scalar(f"Loss/train/{key}", loss, i)

        # resnet/disc loss
        disc_loss = sum([v for k,v in losses_dict.items() if 'fpn' not in k])
        loss_fpn = losses_dict['loss_fpn']
        
        # sum losses 
        loss = disc_loss + loss_fpn 

        # backprop (train end-to-end)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss
        loss = f"[{np.round(disc_loss.item(),5)} + {np.round(loss_fpn.item(),5)} = {np.round(disc_loss.item(),5) +np.round(loss_fpn.item(),5)}]"

        # save if steps
        if i > 0 and i % save_step == 0:
            save_path = f"{checkpoint_dir}/chkpt_{i}.pt"
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, save_path)

    # save final model
    save_path = f"{checkpoint_dir}/chkpt_{i+1}.pt"
    torch.save({
            'epoch': i+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)


# run training code
if __name__ == "__main__":

    # 1. Download DeepForest pre-trained weights
    model = df_main.deepforest()
    model.use_release()
    model.cuda()
    torch.save(model.model.state_dict(), 'df_retinanet.pt')


    # 2. Train model
    train_model('./output/example')