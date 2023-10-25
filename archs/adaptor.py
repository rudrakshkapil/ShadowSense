"""
Script defining UDA Networks (Domain Discriminators) 
"""

import torch
from torch import nn
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F

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
    

""" ------------ Resnet DA architectures (like DA-RetinaNet) --------------- """
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
        x = torch.flatten(x,1)
        x = self.reducer2(x)
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss(x,domain_t,alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss(x,domain_s,alpha=0.25,gamma=2,reduction="mean")
        return loss
    
