"""
Script to obtain predictions of trained model
Evaluation needs to be done separately
"""

import os
import torch
import numpy as np
from skimage.transform import resize

from tqdm import tqdm

from archs.RGBTdetector import RGBxThermalDetector


def save_predictions(model:RGBxThermalDetector, eval_dataloader, exp_dir, mode='combined'):
    """ 
    Prediction function that implementts the proposed Feature Fusion during Inference method
    Background features of both branch FPNs are fused in a weighted manner level-wise and passed to pre-trained detection heads
    """

    save_dir = f"{exp_dir}/predictions_{mode}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # for saving
    img_i = 0
    img_names = os.listdir('data/test/rgb')

    for batch in tqdm(eval_dataloader):
        # get input on gpu
        x_rgb, x_thermal, masks = batch
        x_rgb, x_thermal = x_rgb.cuda(), x_thermal.cuda()

        # extract features
        rgb_feats_res, rgb_feats_fpn, _ = model.rgb_detector(x_rgb)
        thm_feats_res, thm_feats_fpn, _ = model.thermal_detector(model.thermal_prelayer(x_thermal))

        # fusing - get mask (inverted), multiply with thm features
        if 'combined' in mode:
            # get loaded masks as numpy 
            bsz = len(x_rgb)
            masks = masks.cpu().detach().numpy()
            masks = np.logical_not(masks)

            # fusion scaling weights
            scales = [1,1,0.5,0.2,0.2] 
            factor = 5.0

            combined_features = []
            for i,sz in enumerate([64, 32, 16, 8, 4]):
                
                # get mask at current size
                curr_feat = torch.zeros((bsz, 256, sz, sz)).cuda()
                for img_idx in range(len(x_rgb)):
                    mask_rsz = resize(masks[img_idx], (sz,sz))

                    # weighted average of features at background locations
                    new_feat = rgb_feats_fpn[i][img_idx]
                    thm_feat = thm_feats_fpn[i][img_idx]
                    
                    # fuse features and set back in place
                    new_feat[:,mask_rsz] = (new_feat[:,mask_rsz] + thm_feat[:,mask_rsz]*factor*scales[i])/(1.0+factor*scales[i]) # background boosting
                    curr_feat[img_idx, ...] = new_feat

                combined_features.append(curr_feat)

            # pass combined features to pre-trained RetinaNet heads
            detections = model.rgb_detector.predict(x_rgb, combined_features)

        # just RGB or thermal branch predictions
        elif mode == 'rgb':
            detections = model.rgb_detector.predict(x_rgb, rgb_feats_fpn)
        elif mode == 'thermal':
            detections = model.thermal_detector.predict(x_thermal, thm_feats_fpn)

        # save predictions to file
        for detection in detections:
            boxes = detection['boxes'].cpu().detach().numpy()
            scores = detection['scores'].cpu().detach().numpy()

            save_name = f'{save_dir}/{img_names[img_i].split(".")[0]}.txt'
            f = open(save_name, "w")

            for score, box in zip(scores, boxes):
                f.write(f"Tree {score} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n")
            
            img_i += 1
            f.close()





