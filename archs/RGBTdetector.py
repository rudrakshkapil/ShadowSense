"""
Script implementing the proposed Two-branch ShadowSense network
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.functional as F2 

from archs.retinanet import UTDARetinanet
from archs.adaptor import *


class RGBxThermalDetector(nn.Module):
    """
    Proposed ShadowSense model
    """
    def __init__(self, weights_path=None, state_dict=None):
        super().__init__()

        # make detectors (parallel branches)
        num_classes = 1
        self.rgb_detector = UTDARetinanet(num_classes, 'rgb')
        self.thermal_detector = UTDARetinanet(num_classes, 'thermal')

        # pre-layer for thermal to cast 1D to 3 channels
        self.thermal_prelayer = nn.Conv2d(1,3,(1,1))
        
        # load weights (initialise with Deepforest)
        if weights_path is None:
            weights_path = 'df_retinanet.pt'
        if state_dict is None:
            state_dict = torch.load(weights_path)
        self.rgb_detector.load_state_dict(state_dict)
        self.thermal_detector.load_state_dict(state_dict)

        # freeze layers (entire RGB, heads of thermal (which are anyway unused))
        for _, para in self.rgb_detector.named_parameters():
            para.requires_grad = False
            para.grad is None
            
        for name, para in self.thermal_detector.named_parameters():
            if not name.startswith("backbone"):
                para.requires_grad = False
                para.grad = None # NOTE: needed for unfreezing later
        
        # define domain adaptation layers
        self.DA_res3 = DiscriminatorRes3()
        self.DA_res4 = DiscriminatorRes4()
        self.DA_res5 = DiscriminatorRes5()

        
    def return_prelayer_output(self, x_thermal):
        """ Output of 1x1 conv in thermal branch """
        return self.thermal_prelayer(x_thermal)


    def forward(self, x_rgb, x_thermal, alphas, masks, comp_fpn_loss=True):
        """
        Pass x_rgb through rgb_detector;         (get features)
        Pass x_thermal through thermal_detector; (get features)

        Pass features to domain adaptors;
        Compute DAT loss, FG FPN loss

        Returns losses
        """
        # NOTE: features_res: list of 3 tensors (512, 1024, 2048) - (res3, res4, res5)
        # get rgb features and loss 
        rgb_feats_res, rgb_feats_fpn, rgb_head_outputs = self.rgb_detector(x_rgb)
        rgb_ldict = self.compute_DA_loss(rgb_feats_res, rgb_feats_fpn, target_domain=False, alphas=alphas, masks=masks)

        # get thermal features and loss
        thm_input = self.thermal_prelayer(x_thermal)
        thm_feats_res, thm_feats_fpn, thm_head_outputs = self.thermal_detector(thm_input)
        thm_ldict = self.compute_DA_loss(thm_feats_res, thm_feats_fpn, target_domain=True, alphas=alphas, masks=masks)

        # combine losses between modes (average)
        ldict = {}
        for key in rgb_ldict.keys():
            ldict[key] = (rgb_ldict[key] + thm_ldict[key]) / 2.0

        # FG FPN FA 
        ldict.update(self.compute_FG_FPN_feature_alignment_loss(rgb_feats_fpn, thm_feats_fpn, masks))

        # return all losses
        return ldict
    

    def compute_DA_loss(self, features_res, features_fpn, target_domain, alphas, masks):
        """
        Function to compute adaptation loss between three levels of the ResNet & discriminator networks
        NOTE: features_fpn unused - we found that using DA for the FPN perfromed worse
        """
        res3, res4, res5 = features_res
        fpn3, fpn4, fpn5 = features_fpn[:3] # unused for best version of proposed method
        alp3, alp4, alp5 = alphas

        # Compute level-wise adaptation loss (with alpha level for GRL) 
        loss_res3 = self.DA_res3(res3, target_domain, alp3)
        loss_res4 = self.DA_res4(res4, target_domain, alp4)
        loss_res5 = self.DA_res5(res5, target_domain, alp5)
        
        # return
        return {"loss_res3":loss_res3, "loss_res4":loss_res4, "loss_res5":loss_res5}
    

    def compute_FG_FPN_feature_alignment_loss(self, rgb_feats_fpn, thm_feats_fpn, masks):
        """
        Function to compute FG FPN Feature alignment loss using standard MSE 
        """
        loss_dict = {}
        total_loss = 0

        # alignment scales 
        scales = [1,1, 0.5, 0.05, 0.01] 

        # compute loss level-wise
        for i in range(len(rgb_feats_fpn)):
            rgb_feat = rgb_feats_fpn[i].permute((0,2,3,1)).clone()
            thm_feat = thm_feats_fpn[i].permute((0,2,3,1)).clone() 

            mask = F.resize(masks, rgb_feat.shape[1])
            masked_rgb_feats, masked_thm_feats = rgb_feat[mask], thm_feat[mask] # apply foreground mask
            loss_fg = F2.mse_loss(masked_rgb_feats, masked_thm_feats, reduction='mean')

            total_loss += loss_fg * scales[i]
            loss_dict.update({f'loss_fpn_{i}': loss_fg})

        total_loss /= len(rgb_feats_fpn)
        loss_dict.update({'loss_fpn': total_loss})
        return loss_dict
