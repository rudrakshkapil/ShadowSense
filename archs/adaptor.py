import torch
from torch import nn
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F
import torchvision.transforms.functional as Fv

from archs.UNet_parts import *



def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    NOTE: expects input to already be passed through sigmoid()
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    p = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class GradReverse(torch.autograd.Function):
    '''
    Adapted from https://github.com/fpv-iplab/DA-RetinaNet
    '''
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    

    
class FPNLossModule(nn.Module):
    def __init__(self):

        super(FPNLossModule, self).__init__()
        

    def forward(self, x, domain_target = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        # x = self.reducer(x) 
        # print('3', x.shape)
        # x = torch.flatten(x,1)
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss(x,domain_t, alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss(x,domain_s, alpha=0.25,gamma=2,reduction="mean")
        return loss
    
    # def compute_distill_loss_fpn(self, rgb_feats_fpn, thm_feats_fpn, masks):
    #     for i in range(len(thm_feats_fpn)):
    #         thm_feats_fpn[i] = GradReverse.apply(thm_feats_fpn[i], 1)

    #     total_loss = 0

    #     # NOTE:*= for weighted 
    #     # masks *= 2
    #     # masks[masks > 0] = 1.0
    #     # masks = torch.clip(masks, min=0.0, max=1.0)

    #     # NOTE: scales = [1,1, 0.5, 0.05, 0.01]
    #     scales = [1,1,1,1,1]
                                      
        
    #     for i in range(len(rgb_feats_fpn)):
    #         rgb_feat = rgb_feats_fpn[i].permute((0,2,3,1))
    #         thm_feat = thm_feats_fpn[i].permute((0,2,3,1))

    #         mask = F.resize(masks, rgb_feat.shape[1])
    #         masked_rgb_feats, masked_thm_feats = rgb_feat[mask], thm_feat[mask] # orig
    #         loss_fg = F2.mse_loss(masked_rgb_feats, masked_thm_feats, reduction='mean')

    #         total_loss += loss_fg

    #         # NOTE: for weighted masks:
    #         # mask_weights = F.resize(masks, rgb_feat.shape[1]) 
    #         # mask = torch.zeros_like(mask_weights, dtype=torch.bool)
    #         # mask[mask_weights > 0] = 1
    #         # mask_weights = mask_weights.unsqueeze(3)

    #         # rgb_feat *= mask_weights
    #         # thm_feat *= mask_weights

    #         # masked_rgb_feats, masked_thm_feats = rgb_feat[mask], thm_feat[mask] # orig
    #         # loss_fg = F2.mse_loss(masked_rgb_feats, masked_thm_feats, reduction='mean')
    #         # total_loss += loss_fg * scales[i] # NOTE: for scaling


    #         # NOTE; different averaging for scales - old weighting basically
    #         # masked_rgb_feats, masked_thm_feats = rgb_feat[mask], thm_feat[mask] # orig
    #         # loss_fg = F2.mse_loss(rgb_feat, thm_feat, reduction='sum')
    #         # print(loss_fg)
    #         # fg_counts = torch.count_nonzero(mask_weights)
    #         # weight = (fg_counts*10**(torch.log(rgb_feat.shape[1])))
    #         # loss_fg /= (weight*rgb_feat.shape[0])
    #         # total_loss += loss_fg
    #         # print(fg_counts, weight, loss_fg)


    #         # NOTE: contrastive
    #         # mask_bg = torch.logical_not(mask)
    #         # masked_rgb_feats_bg, masked_thm_feats_bg = rgb_feat[mask_bg], thm_feat[mask_bg]
    #         # diff = torch.square(masked_rgb_feats_bg - masked_thm_feats_bg)
    #         # diff = 1.0-diff 
    #         # diff[diff<0] = 0
    #         # loss_bg =
    #         # 
    #         #   def compute_distill_loss_fpn(self, rgb_feats_fpn, thm_feats_fpn, masks):

    #     total_loss = 0

    #     # NOTE:*= for weighted 
    #     # masks *= 2
    #     # masks[masks > 0] = 1.0
    #     # masks = torch.clip(masks, min=0.0, max=1.0)

    #     # NOTE: scales = [1,1, 0.5, 0.05, 0.01]
    #     scales = [1,1,1,1,1]
                                      
        
    #     for i in range(len(rgb_feats_fpn)):
    #         rgb_feat = rgb_feats_fpn[i].permute((0,2,3,1))
    #         thm_feat = thm_feats_fpn[i].permute((0,2,3,1))

    #         mask = F.resize(masks, rgb_feat.shape[1])
    #         masked_rgb_feats, masked_thm_feats = rgb_feat[mask], thm_feat[mask] # orig
    #         loss_fg = F2.mse_loss(masked_rgb_feats, masked_thm_feats, reduction='mean')

    #         total_loss += loss_fg

            

    #     total_loss /= len(rgb_feats_fpn)
    #     return {'loss_fpn': total_loss} 

            
    

# ------------ Resnet DA like DA-RetinaNet
class DiscriminatorRes3(nn.Module):

    def __init__(self):

        super(DiscriminatorRes3, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = (1, 1) ,bias = False), # 64x64 -> 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size = (1, 1) ,bias = False), # 64x64 -> 
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 1, kernel_size=(1, 1), bias = False)
        ).cuda()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_target = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        # print('3', x.shape)
        x = torch.flatten(x,1)
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss(x,domain_t, alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss(x,domain_s, alpha=0.25,gamma=2,reduction="mean")
        return loss

class DiscriminatorRes4(nn.Module):

    def __init__(self):

        super(DiscriminatorRes4, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(512, 128, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(128, 128, kernel_size = (3, 3), stride = 2, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).cuda()
        self.reducer2 = nn.Linear(128, 1, bias = False ).cuda()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_target = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        # print('4', x.shape)
        # print(torch.amax(x), torch.amin(x))
        x = torch.flatten(x,1)
        x = self.reducer2(x)
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss(x,domain_t,alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss(x,domain_s,alpha=0.25,gamma=2,reduction="mean")
        return loss
        
class DiscriminatorRes5(nn.Module):

    def __init__(self):
        super(DiscriminatorRes5, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(1024, 256, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(256, 256, kernel_size = (3, 3), stride = 2, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).cuda()
        self.reducer2 = nn.Sequential(
            nn.Linear(256, 128, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(128, 1, bias= False)
        ).cuda() 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_target = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        # print('5', x.shape)
        # print(torch.amax(x), torch.amin(x))
        x = torch.flatten(x,1)
        x = self.reducer2(x)
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss(x,domain_t,alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss(x,domain_s,alpha=0.25,gamma=2,reduction="mean")
        return loss
    

# ----------- FFDA with only upconv + Image-level discrimination -------------
class FFDA3_withImage(nn.Module):
    def __init__(self):
        super(FFDA3_withImage, self).__init__()

        # 1. 
        self.upscaler = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size = (2,2), stride=2, bias = False), # 512x64x64 -> 128x128x128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, bias=False),                   # -> 64x126x126
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size = (2,2), stride=2, bias = False),   # -> 32x252x252
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, bias=False),                    # -> 16x250x250
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size = (2,2), stride=2, bias = False),    # -> 1x500x500
            nn.Sigmoid()
        ).cuda()

        self.reducer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = (1, 1) ,bias = False), # 64x64 -> 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size = (1, 1) ,bias = False), # 64x64 -> 
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 1, kernel_size=(1, 1), bias = False)
        ).cuda()

        self.level_name = 'res3'

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def grad_reverse(self, x, alpha):
        return GradReverse.apply(x, alpha)
    
    def pixel_loss(self, x_orig, masks, domain_target=False):
        '''compute normalized foreground BCE for x given masks (both bszx500x500), depending on RGB or thermal'''
        
        # get counts for normalizing
        fg_counts = torch.sum(masks, axis=(1,2))

        # NOTE: old way (making a clone)
        # get bg indices, and set as target (don't want to calculate loss with those pixels)
        # neg_masks = torch.logical_not(masks)
        # x = x_orig.clone()
        # x[neg_masks] = 1 if domain_target else 0
        # x = torch.flatten(x,1)

        # flatten to pass to loss function
        x2 = x_orig[masks]        
        
        # compute loss
        if domain_target:
            domain_t = torch.ones(x2.size()).float().cuda()
            loss = focal_loss(x2,domain_t, alpha=0.25,gamma=2)  
        else:
            domain_s = torch.zeros(x2.size()).float().cuda()
            loss = focal_loss(x2,domain_s, alpha=0.25,gamma=2)

        # reduce loss to single value: bsz x 250000  (sum)-> bsz x 1  (w.avg)-> 1
        mean = torch.mean(loss)
        # print(mean)
        # print(loss.shape)
        # loss = torch.sum(loss, axis=1)
        # print(loss.shape)
        # loss = torch.mean(loss / fg_counts)
        # exit()
        return mean
    
    def img_loss(self, x, domain_target):
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss(x,domain_t, alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss(x,domain_s, alpha=0.25,gamma=2,reduction="mean")
        return loss
    

    def forward(self, x_in, masks, domain_target = False, alpha = 1):
        x_rev = self.grad_reverse(x_in, alpha)

        # NOTE: for without upscaler
        print(masks)
        masks = Fv.resize(masks, x_in.shape[2:])
        pixel_loss = self.pixel_loss(x_in.permute((0,2,3,1)), masks, domain_target)
        
        # NOTE: with upscaler
        # x = self.upscaler(x_rev)
        # x = x.squeeze(1)     # bsz x 500 x 500 (same as masks)  
        # pixel_loss = self.pixel_loss(x_in, masks, domain_target)                 

        x = self.reducer(x_rev)
        x = torch.flatten(x,1)
        img_loss = self.img_loss(x, domain_target)

        return {f'pixel_loss_{self.level_name}':pixel_loss, f'img_loss_{self.level_name}':img_loss}
        

class FFDA4_withImage(FFDA3_withImage):
    def __init__(self):
        super(FFDA4_withImage, self).__init__()
        self.level_name = 'res4'

        # this new module just converts 1024x32x32 to 512x64x64, then we can just use the upsampler defined in FFDA3
        self.pre_upscaler = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=(2,2), stride=2, bias=False),
            nn.ReLU(),
        )

        self.reducer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(512, 128, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(128, 128, kernel_size = (3, 3), stride = 2, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).cuda()
        self.reducer2 = nn.Linear(128, 1, bias = False ).cuda()

        self.init_weights()



    def forward(self, x_in, masks, domain_target = False, alpha = 1):
        x_rev = self.grad_reverse(x_in, alpha)

        # NOTE: for without upscaler
        masks = Fv.resize(masks, x_in.shape[2:])
        pixel_loss = self.pixel_loss(x_in.permute((0,2,3,1)), masks, domain_target)
        
        # NOTE: with upscaler
        # x = self.pre_upscaler(x_rev)
        # x = self.upscaler(x) 
        # x = x.squeeze(1)     # bsz x 500 x 500 (same as masks)
        # pixel_loss = self.pixel_loss(x, masks, domain_target)

        x = self.reducer(x_rev) 
        x = torch.flatten(x,1)
        x = self.reducer2(x)
        img_loss = self.img_loss(x, domain_target)

        return {f'pixel_loss_{self.level_name}':pixel_loss, f'img_loss_{self.level_name}':img_loss}



class FFDA5_withImage(FFDA3_withImage):
    def __init__(self):
        super(FFDA5_withImage, self).__init__()
        self.level_name = 'res5'

        # NOTE: just added one more ConvTranspose 2d to what's there in FFDA4
        # this new module just converts 2048x16x16 to 512x64x64, then we can just use the upsampler defined in FFDA3
        self.pre_upscaler = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=(2,2), stride=2, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=(2,2), stride=2, bias=False),
            nn.ReLU(),
        )

        self.reducer = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(1024, 256, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(256, 256, kernel_size = (3, 3), stride = 2, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).cuda()
        self.reducer2 = nn.Sequential(
            nn.Linear(256, 128, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(128, 1, bias= False)
        ).cuda() 

        self.init_weights()

    def forward(self, x_in, masks, domain_target = False, alpha = 1):
        x_rev = self.grad_reverse(x_in, alpha)

        # NOTE: for without upscaler
        masks = Fv.resize(masks, x_in.shape[2:])
        pixel_loss = self.pixel_loss(x_in.permute((0,2,3,1)), masks, domain_target)
        
        # NOTE: with upscaler
        # x = self.pre_upscaler(x_rev)
        # x = self.upscaler(x) 
        # x = x.squeeze(1)     # bsz x 500 x 500 (same as masks)
        # pixel_loss = self.pixel_loss(x, masks, domain_target)

        x = self.reducer(x_rev) 
        x = torch.flatten(x,1)
        x = self.reducer2(x)
        img_loss = self.img_loss(x, domain_target)

        return {f'pixel_loss_{self.level_name}':pixel_loss, f'img_loss_{self.level_name}':img_loss}




# ----------- FFDA using UNet-like layers
class FFDA_UNet(nn.Module):
    def __init__(self):
        super(FFDA_UNet, self).__init__()

        self.up1 = (Up(2048, 1024))
        self.up2 = (Up(1024, 512))

        
        self.up3 = nn.Sequential(
            Up(512, 128, mid_channels=256, single=True, reduction=True), # 512x64x64 -> 256x128x128
            Up(128,  32, mid_channels=64, single=True, reduction=True),  # -> 128x256x256
            Up(32, 16, mid_channels=16, single=True, reduction=False),
            OutConv(16, 1),
            nn.Sigmoid()
        )

        # 1. 
        self.upscaler = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size = (2,2), stride=2, bias = False), # 512x64x64 -> 128x128x128
            nn.Conv2d(128, 64, kernel_size=3, stride=1, bias=False),                   # -> 64x126x126
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size = (2,2), stride=2, bias = False),   # -> 32x252x252
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, bias=False),                    # -> 16x250x250
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size = (2,2), stride=2, bias = False),    # -> 1x500x500
            nn.Sigmoid()
        ).cuda()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def grad_reverse(self, x, alpha):
        return GradReverse.apply(x, alpha)
    
    def compute_loss(self, x_orig, masks, domain_target=False, alpha = 1):
        '''compute normalized foreground BCE for x given masks (both bszx500x500), depending on RGB or thermal'''
        
        # get counts for normalizing
        fg_counts = torch.sum(masks, axis=(1,2))

        # get bg indices, and set as target (don't want to calculate loss with those pixels)
        neg_masks = torch.logical_not(masks)
        x = x_orig.clone()
        x[neg_masks] = 1 if domain_target else 0

        # flatten to pass to loss function
        x = torch.flatten(x,1)
        
        # compute loss
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = focal_loss(x,domain_t, alpha=0.25,gamma=2)
            
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = focal_loss(x,domain_s, alpha=0.25,gamma=2)

        # reduce loss to single value: bsz x 250000  (sum)-> bsz x 1  (w.avg)-> 1
        loss = torch.sum(loss, axis=1)
        loss = torch.mean(loss / fg_counts)
        return loss

    def forward(self, x3, x4, x5, masks, domain_target = False, alpha = 1):
        x3 = self.grad_reverse(x3, alpha)
        x4 = self.grad_reverse(x4, alpha)
        x5 = self.grad_reverse(x5, alpha)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x)
        x = x.squeeze(1)     # bsz x 500 x 500 (same as masks)            
        
        return self.compute_loss(x, masks, domain_target, alpha)
        



# ----------- FFDA with only upconv -------------
class FFDA3(nn.Module):
    def __init__(self):
        super(FFDA3, self).__init__()

        # 1. 
        self.upscaler = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size = (2,2), stride=2, bias = False), # 512x64x64 -> 128x128x128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, bias=False),                   # -> 64x126x126
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size = (2,2), stride=2, bias = False),   # -> 32x252x252
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, bias=False),                    # -> 16x250x250
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size = (2,2), stride=2, bias = False),    # -> 1x500x500
            nn.Sigmoid()
        ).cuda()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def grad_reverse(self, x, alpha):
        return GradReverse.apply(x, alpha)
    
    def compute_loss(self, x_orig, masks, domain_target=False, alpha = 1):
        '''compute normalized foreground BCE for x given masks (both bszx500x500), depending on RGB or thermal'''
        
        # get counts for normalizing
        fg_counts = torch.sum(masks, axis=(1,2))

        # get bg indices, and set as target (don't want to calculate loss with those pixels)
        neg_masks = torch.logical_not(masks)
        x = x_orig.clone()
        x[neg_masks] = 1 if domain_target else 0

        # flatten to pass to loss function
        x = torch.flatten(x,1)
        
        # compute loss
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = focal_loss(x,domain_t, alpha=0.25,gamma=2)
            
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = focal_loss(x,domain_s, alpha=0.25,gamma=2)

        # reduce loss to single value: bsz x 250000  (sum)-> bsz x 1  (w.avg)-> 1
        loss = torch.sum(loss, axis=1)
        loss = torch.mean(loss / fg_counts)
        return loss

    def forward(self, x, masks, domain_target = False, alpha = 1):
        x = self.grad_reverse(x, alpha)
        x = self.upscaler(x)
        x = x.squeeze(1)     # bsz x 500 x 500 (same as masks)            
        
        return self.compute_loss(x, masks, domain_target, alpha)
        

class FFDA4(FFDA3):
    def __init__(self):
        super(FFDA4, self).__init__()

        # this new module just converts 1024x32x32 to 512x64x64, then we can just use the upsampler defined in FFDA3
        self.pre_upscaler = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=(2,2), stride=2, bias=False),
            nn.ReLU(),
        )
        self.init_weights()

    def forward(self, x, masks, domain_target = False, alpha = 1):
        x = self.grad_reverse(x, alpha)
        x = self.pre_upscaler(x)
        x = self.upscaler(x) 
        x = x.squeeze(1)     # bsz x 500 x 500 (same as masks)

        return self.compute_loss(x, masks, domain_target, alpha)


class FFDA5(FFDA3):
    def __init__(self):
        super(FFDA5, self).__init__()

        # NOTE: just added one more ConvTranspose 2d to what's there in FFDA4
        # this new module just converts 2048x16x16 to 512x64x64, then we can just use the upsampler defined in FFDA3
        self.pre_upscaler = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=(2,2), stride=2, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=(2,2), stride=2, bias=False),
            nn.ReLU(),
        )
        self.init_weights()

    def forward(self, x, masks, domain_target = False, alpha = 1):
        x = self.grad_reverse(x, alpha)
        x = self.pre_upscaler(x)
        x = self.upscaler(x) 
        x = x.squeeze(1)     # bsz x 500 x 500 (same as masks)

        return self.compute_loss(x, masks, domain_target, alpha)




# -------------- FPN DA ---------------
# class DiscriminatorFPN(nn.Module):

#     def __init__(self):

#         super(DiscriminatorFPN, self).__init__()
#         self.reducer = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size = (3,3), bias = False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace = True),
#             nn.Dropout(),
#             nn.Conv2d(128, 64, kernel_size = (3, 3), bias = False),  
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace = True),
#             nn.Dropout(),
#             nn.AdaptiveAvgPool2d((1, 1)),
#         ).cuda()
#         self.reducer2 = nn.Linear(64, 1, bias = False).cuda()
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

#     def forward(self, x, domain_target = False, alpha = 1):
#         x = GradReverse.apply(x, alpha)
#         x = self.reducer(x) 
#         x = torch.flatten(x,1)
#         x = self.reducer2(x)
#         if domain_target:
#             domain_t = torch.ones(x.size()).float().cuda()
#             loss = sigmoid_focal_loss(x,domain_t, alpha=0.25,gamma=2,reduction="mean")
#         else:
#             domain_s = torch.zeros(x.size()).float().cuda()
#             loss = sigmoid_focal_loss(x,domain_s, alpha=0.25,gamma=2,reduction="mean")
#         return loss
    

# class DiscriminatorFPN3(nn.Module):

#     def __init__(self):

#         super(DiscriminatorFPN3, self).__init__()
#         self.reducer = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size = (1, 1) ,bias = False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 128, kernel_size = (1, 1) ,bias = False),  
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(128, 1, kernel_size=(1, 1), bias = False)
#         ).cuda()
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

#     def forward(self, x, domain_target = False, alpha = 1):
#         x = GradReverse.apply(x, alpha)
#         x = self.reducer(x) 
#         x = torch.flatten(x,1)
#         if domain_target:
#             domain_t = torch.ones(x.size()).float().cuda()
#             loss = sigmoid_focal_loss(x,domain_t, alpha=0.25,gamma=2,reduction="mean")
#         else:
#             domain_s = torch.zeros(x.size()).float().cuda()
#             loss = sigmoid_focal_loss(x,domain_s, alpha=0.25,gamma=2,reduction="mean")
#         return loss

# class DiscriminatorFPN4(nn.Module):

#     def __init__(self):

#         super(DiscriminatorFPN4, self).__init__()
#         self.reducer = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size = (3, 3), stride = 2, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace = True),
#             nn.Dropout(),
#             nn.Conv2d(128, 64, kernel_size = (3, 3), stride = 2, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace = True),
#             nn.Dropout(),
#             nn.Conv2d(64, 64, kernel_size = (3, 3), stride = 2, bias=False), 
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace = True),
#             nn.Dropout(),
#             nn.AdaptiveAvgPool2d((1, 1)),
#         ).cuda()
#         self.reducer2 = nn.Linear(64, 1, bias = False ).cuda()

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

#     def forward(self, x, domain_target = False, alpha = 1):
#         x = GradReverse.apply(x, alpha)
#         x = self.reducer(x) 
#         x = torch.flatten(x,1)
#         x = self.reducer2(x)
#         if domain_target:
#             domain_t = torch.ones(x.size()).float().cuda()
#             loss = sigmoid_focal_loss(x,domain_t,alpha=0.25,gamma=2,reduction="mean")
#         else:
#             domain_s = torch.zeros(x.size()).float().cuda()
#             loss = sigmoid_focal_loss(x,domain_s,alpha=0.25,gamma=2,reduction="mean")
#         return loss
        
# class DiscriminatorFPN5(nn.Module):

    # def __init__(self):
    #     super(DiscriminatorFPN5, self).__init__()
    #     self.reducer = nn.Sequential(
    #         nn.Conv2d(256, 128, kernel_size = (3, 3), stride = 2, bias=False),
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(inplace = True),
    #         nn.Dropout(),
    #         nn.Conv2d(128, 64, kernel_size = (3, 3), stride = 2, bias=False),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(inplace = True),
    #         nn.Dropout(),
    #         nn.Conv2d(64, 64, kernel_size = (3, 3), stride = 2, bias=False), 
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(inplace = True),
    #         nn.Dropout(),
    #         nn.AdaptiveAvgPool2d((1, 1)),
    #     ).cuda()
    #     self.reducer2 = nn.Sequential(
    #         nn.Linear(64, 32, bias = False),
    #         nn.BatchNorm1d(32),
    #         nn.ReLU(inplace = True),
    #         nn.Dropout(),
    #         nn.Linear(32, 1, bias= False)
    #     ).cuda() 

    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    # def forward(self, x, domain_target = False, alpha = 1):
    #     x = GradReverse.apply(x, alpha)
    #     x = self.reducer(x) 
    #     x = torch.flatten(x,1)
    #     x = self.reducer2(x)
    #     if domain_target:
    #         domain_t = torch.ones(x.size()).float().cuda()
    #         loss = sigmoid_focal_loss(x,domain_t,alpha=0.25,gamma=2,reduction="mean")
    #     else:
    #         domain_s = torch.zeros(x.size()).float().cuda()
    #         loss = sigmoid_focal_loss(x,domain_s,alpha=0.25,gamma=2,reduction="mean")
    #     return loss