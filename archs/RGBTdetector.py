import torch
import torch.nn as nn
from skimage.transform import resize

from archs.retinanet import UTDARetinanet
from archs.adaptor import *

import torchvision.transforms.functional as F
import torch.nn.functional as F2 

class RGBxThermalDetector(nn.Module):
    def __init__(self, weights_path=None, state_dict=None):
        super().__init__()

        # make detectors
        num_classes = 1
        self.rgb_detector = UTDARetinanet(num_classes, 'rgb')
        self.thermal_detector = UTDARetinanet(num_classes, 'thermal')

        # pre-layer for thermal to cast to 3 channels
        self.thermal_prelayer = nn.Conv2d(1,3,(1,1))
        
        # load weights
        if weights_path is None:
            weights_path = 'df_retinanet.pt'
        # with open(weights_path) as f:
        if state_dict is None:
            state_dict = torch.load(weights_path)
        self.rgb_detector.load_state_dict(state_dict)
        self.thermal_detector.load_state_dict(state_dict)

        # freeze layers (entire RGB, FPN onwards of thermal)
        for _, para in self.rgb_detector.named_parameters():
            para.requires_grad = False
            para.grad is None
            

        for name, para in self.thermal_detector.named_parameters():
            if not name.startswith("backbone"): # NOTE: 7. .body => FPN also frozen
                para.requires_grad = False
                para.grad = None # NOTE: needed for unfreezing later
        
        ## make adaptiation layers
        # 1. DA-RetinaNet: # NOTE: best
        self.DA_res3 = DiscriminatorRes3()
        self.DA_res4 = DiscriminatorRes4()
        self.DA_res5 = DiscriminatorRes5()

        # 2. FPN (tried 2 diff ways)
        # self.DA_FPN3 = DiscriminatorFPN()
        # self.DA_FPN4 = DiscriminatorFPN()
        # self.DA_FPN5 = DiscriminatorFPN()
        # self.DA_FPN3 = DiscriminatorFPN3()
        # self.DA_FPN4 = DiscriminatorFPN4()
        # self.DA_FPN5 = DiscriminatorFPN5()

        # 3. FFDA
        # self.DA_res3 = FFDA3()
        # self.DA_res4 = FFDA4()
        # self.DA_res5 = FFDA5()

        # 4. FFDA with UNet like strcuture
        # self.DA_res_unet = FFDA_UNet()

        # 5. fFDa with Image level loss as well
        # self.DA_res3 = FFDA3_withImage()
        # self.DA_res4 = FFDA4_withImage()
        # self.DA_res5 = FFDA5_withImage()

    def return_prelayer_output(self, x_thermal):
        return self.thermal_prelayer(x_thermal)


    def forward(self, x_rgb, x_thermal, alphas, masks, comp_fpn_loss=True):
        """
        Pass x_rgb through rgb_detector;         (get features)
        Pass x_thermal through thermal_detector; (get features)

        Pass features to domain adaptors;
        Backprop (compute loss);

        Returns loss
        """
        # NOTE: features_res: list of 3 tensors (512, 1024, 2048) - (res3, res4, res5)

        ## Try 1. mask input
        # masks = masks.unsqueeze(1)
        # x_rgb = x_rgb * masks
        # x_thermal = x_thermal * masks

        # get rgb features and loss NOTE: turn on 
        rgb_feats_res, rgb_feats_fpn, rgb_head_outputs = self.rgb_detector(x_rgb)
        rgb_ldict = self.compute_DA_loss(rgb_feats_res, rgb_feats_fpn, target_domain=False, alphas=alphas, masks=masks)

        # get thermal features and loss
        thm_input = self.thermal_prelayer(x_thermal)
        thm_feats_res, thm_feats_fpn, thm_head_outputs = self.thermal_detector(thm_input)
        thm_ldict = self.compute_DA_loss(thm_feats_res, thm_feats_fpn, target_domain=True, alphas=alphas, masks=masks)

        # 1,4. combine losses between modes
        ldict = {}
        for key in rgb_ldict.keys():
            ldict[key] = (rgb_ldict[key] + thm_ldict[key]) / 2.0

        # ldict["loss_res3"] = (rgb_ldict["loss_res3"] + thm_ldict["loss_res3"]) / 2.0 
        # ldict["loss_res4"] = (rgb_ldict["loss_res4"] + thm_ldict["loss_res4"]) / 2.0 
        # ldict["loss_res5"] = (rgb_ldict["loss_res5"] + thm_ldict["loss_res5"]) / 2.0 
        # ldict["loss_res_ffdaUnet"] = (rgb_ldict["loss_res_ffdaUnet"] + thm_ldict["loss_res_ffdaUnet"]) / 2.0 

        # 6. distillation loss:
        # ldict.update(self.compute_distill_loss(rgb_head_outputs, thm_head_outputs, masks))

        # 7. distillation loss for FPN
        if ldict is None:
            ldict = {}
        if comp_fpn_loss:
            ldict.update(self.compute_distill_loss_fpn(rgb_feats_fpn, thm_feats_fpn, masks))


        # 3. 
        # ldict["loss_fpn3"] = (rgb_ldict["loss_fpn3"] + thm_ldict["loss_fpn3"]) / 2.0 
        # ldict["loss_fpn4"] = (rgb_ldict["loss_fpn4"] + thm_ldict["loss_fpn4"]) / 2.0 
        # ldict["loss_fpn5"] = (rgb_ldict["loss_fpn5"] + thm_ldict["loss_fpn5"]) / 2.0 

        return ldict
    
    def compute_DA_loss(self, features_res, features_fpn, target_domain, alphas, masks):
        res3, res4, res5 = features_res
        fpn3, fpn4, fpn5 = features_fpn[:3]
        alp3, alp4, alp5 = alphas

        # 1.  DA-retinaet loss (only resnet) 
        loss_res3 = self.DA_res3(res3, target_domain, alp3)
        loss_res4 = self.DA_res4(res4, target_domain, alp4)
        loss_res5 = self.DA_res5(res5, target_domain, alp5)
        

        # # 2. mask features before sending to loss
        # masks = masks.unsqueeze(1)
        # masks3 = F.resize(masks, res3.shape[-1])
        # masks4 = F.resize(masks, res4.shape[-1])
        # masks5 = F.resize(masks, res5.shape[-1])
        
        # res3_masked = res3 * masks3
        # res4_masked = res4 * masks4
        # res5_masked = res5 * masks5

        # loss_res3 = self.DA_res3(res3_masked, target_domain, alp3)
        # loss_res4 = self.DA_res4(res4_masked, target_domain, alp4)
        # loss_res5 = self.DA_res5(res5_masked, target_domain, alp5)


        # 3. FPN no mask - not good => leave FPN frozen?
        # loss_fpn3 = self.DA_FPN3(fpn3, target_domain, alp3)
        # loss_fpn4 = self.DA_FPN4(fpn4, target_domain, alp4)
        # loss_fpn5 = self.DA_FPN5(fpn5, target_domain, alp5)


        # 4. FFDA - masked resnet loss (cross entropy loss) -- different discriminator (should output binary map)
        # loss_res3 = self.DA_res3(res3, masks, target_domain, alp3)
        # loss_res4 = self.DA_res4(res4, masks, target_domain, alp4)
        # loss_res5 = self.DA_res5(res5, masks, target_domain, alp5)

        # 5. FFDA with UNet structure
        # loss = self.DA_res_unet(res3, res4, res5, masks, target_domain, alp3)

        # 6. FFDA (pixel level loss) + Image level loss as well (like DA-RetinaNet)
        # loss = {}
        # loss.update(self.DA_res3(res3, masks, domain_target=target_domain, alpha=alp3))
        # loss.update(self.DA_res4(res4, masks, domain_target=target_domain, alpha=alp4))
        # loss.update(self.DA_res5(res5, masks, domain_target=target_domain, alpha=alp5))

 

        # return loss
        # return {"loss_res_ffdaUnet": loss}
        return {"loss_res3":loss_res3, "loss_res4":loss_res4, "loss_res5":loss_res5}
        # return {"loss_res3":loss_res3, "loss_res4":loss_res4, "loss_res5":loss_res5,
        #         "loss_fpn3":loss_fpn3, "loss_fpn4":loss_fpn4, "loss_fpn5":loss_fpn5}

    
    def compute_distill_loss_heads(self, rgb_head_outputs, thm_head_outputs, masks):

        # first cls_logits, then other 
        all_losses = {}
        rgb_feats, thm_feats = {}, {}
        for k in rgb_head_outputs.keys():
            rgb_feats = rgb_head_outputs[k][1]
            thm_feats = thm_head_outputs[k][1]

            total_loss = 0
            
            for i in range(len(rgb_feats)):
                rgb_feat = rgb_feats[i].permute((0,2,3,1))
                thm_feat = thm_feats[i].permute((0,2,3,1))
                mask = F.resize(masks, rgb_feat.shape[1])

                masked_rgb_feats, masked_thm_feats = rgb_feat[mask], thm_feat[mask]
                loss = F2.mse_loss(masked_rgb_feats, masked_thm_feats, reduction='mean')
                total_loss += loss
            all_losses[f'loss_distill_{k}'] = total_loss/len(rgb_feats)

        return all_losses

     
    def compute_distill_loss_fpn(self, rgb_feats_fpn, thm_feats_fpn, masks):

        total_loss = 0

        # NOTE:*= for weighted 
        # masks *= 2
        # masks[masks > 0] = 1.0
        # masks = torch.clip(masks, min=0.0, max=1.0)

        # NOTE: ablation
        scales = [1,1, 0.5, 0.05, 0.01] # NOTE: best
        # scales = [1,0.5, 0.1, 0.01, 0.005] 
        # scales = [1,1,0.75,0.5,0.25] 
        # scales = [1,1,1,1,1]
        # scales = [0.01, 0.05, 0.5, 1, 1]
        # scales = [0.005, 0.01, 0.1, 0.5, 1] 
        # scales = [0.25,0.5,0.75,1,1] 
        loss_dict = {}
        
        for i in range(len(rgb_feats_fpn)):
            rgb_feat = rgb_feats_fpn[i].permute((0,2,3,1)).clone()
            thm_feat = thm_feats_fpn[i].permute((0,2,3,1)).clone() 

            mask = F.resize(masks, rgb_feat.shape[1])
            masked_rgb_feats, masked_thm_feats = rgb_feat[mask], thm_feat[mask] # orig
            # masked_rgb_feats, masked_thm_feats = rgb_feat, thm_feat #[mask], thm_feat[mask] # orig
            loss_fg = F2.mse_loss(masked_rgb_feats, masked_thm_feats, reduction='mean')

            total_loss += loss_fg * scales[i]

            loss_dict.update({f'loss_fpn_{i}': loss_fg})

            # # NOTE: for weighted masks:
            # mask_weights = F.resize(masks, rgb_feat.shape[1]) 
            # mask = torch.zeros_like(mask_weights, dtype=torch.bool)
            # mask[mask_weights > 0] = 1
            # mask_weights = mask_weights.unsqueeze(3)

            # rgb_feat *= mask_weights
            # thm_feat *= mask_weights

            # masked_rgb_feats, masked_thm_feats = rgb_feat[mask], thm_feat[mask] # orig
            # loss_fg = F2.mse_loss(masked_rgb_feats, masked_thm_feats, reduction='mean')
            # total_loss += loss_fg * scales[i] # NOTE: for scaling

            # # update loss
            # loss_dict.update({f'loss_fpn_{i}': loss_fg})


            # NOTE; different averaging for scales - old weighting basically
            # masked_rgb_feats, masked_thm_feats = rgb_feat[mask], thm_feat[mask] # orig
            # loss_fg = F2.mse_loss(rgb_feat, thm_feat, reduction='sum')
            # print(loss_fg)
            # fg_counts = torch.count_nonzero(mask_weights)
            # weight = (fg_counts*10**(torch.log(rgb_feat.shape[1])))
            # loss_fg /= (weight*rgb_feat.shape[0])
            # total_loss += loss_fg
            # print(fg_counts, weight, loss_fg)


            # NOTE: contrastive
            # mask_bg = torch.logical_not(mask)
            # masked_rgb_feats_bg, masked_thm_feats_bg = rgb_feat[mask_bg], thm_feat[mask_bg]
            # diff = torch.square(masked_rgb_feats_bg - masked_thm_feats_bg)
            # diff = 1.0-diff 
            # diff[diff<0] = 0
            # loss_bg = torch.mean(diff)

            # total_loss += (loss_fg + loss_bg)

        total_loss /= len(rgb_feats_fpn)
        loss_dict.update({'loss_fpn': total_loss})
        return loss_dict

                




    


