"""
Script defining network architecture for Triple Domain Adaptation RetinaNet 
i.e., one branch of the proposed RGB-thermal model
"""

import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.models.detection.retinanet import RetinaNet

from archs.backbone import create_backbone


class UTDARetinanet(RetinaNet):
    """
    Unsupervised Triple Domain Adaptation RetinaNet architecture.
    Allows having DA modules at ResNet layers, FPN layers, and head layers

    Same as PyTorch Retinanet, but with different forward() method that: 
        - returns intermediate layer features for use in DA modules
        - allows backbone.forward() to resnet features & FPN features
        - does not take any targets (only extracts features)
    """
    def __init__(self, num_classes, mode):
        # create bacbone with FPN and create RetinaNet
        backbone = create_backbone()
        # backbone.cuda()

        # stats of RT-Trees dataset
        if mode == 'rgb':
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
            
        elif mode == 'thermal':
            image_mean = [13.7140]
            image_std = [1.8340]
    
        # RetinaNet params
        score_thresh = 0.1 
        nms_thresh = 0.15 

        # call parent init function with passed params
        super().__init__(backbone, num_classes, 
                         image_mean=image_mean, image_std=image_std, 
                         score_thresh=score_thresh, nms_thresh=nms_thresh,
                         min_size=500,max_size=500, head=None) 
        

    def forward(self, images):
        '''
        Input: transformed images (tensors)
        '''
        # transform images
        images, _ = self.transform(images)
    
        # get the features from the backbone (ResNet, FPN)
        features_res, features_fpn = self.backbone(images.tensors)
        if isinstance(features_fpn, torch.Tensor):
            features_fpn = OrderedDict([("0", features_fpn)])
        if isinstance(features_res, torch.Tensor):
            features_res = OrderedDict([("0", features_res)])

        # get feature maps as lists (from tensors)
        features_fpn = list(features_fpn.values())
        features_res = list(features_res.values())

        # compute the retinanet heads outputs using the features 
        head_outputs = self.head(features_fpn)

        # return features and head outputs
        return features_res, features_fpn, head_outputs
    

    def predict(self, images, features=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            features (optional): Needs to be given for thermal (1 channel). if given, assumed 
                                 to be the fused features from RGB and thermal and used to perform prediction. 
        """

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))


        # if (fused) features not given, need to find ourselves
        if features is None:
            # pass through layers. 'features' are from fpn (2nd arg)
            _, features, head_outputs = self.forward(images)
            images, _ = self.transform(images)
        else:
            images, _ = self.transform(images)
            head_outputs = self.head(features) 

        # if using distill loss, get rid of all_features (second return value in head outputs)
        if self.distill:
            for k,v in head_outputs.items():
                head_outputs[k] = v[0]

        # create the set of anchors
        # images = images / 255
        # images = ImageList(images, [(original_image_sizes[0][0],original_image_sizes[0][1])]*len(images))
        # images, _ = self.transform(images)
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []

        # recover level sizes
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        HW = 0
        for v in num_anchors_per_level:
            HW += v
        HWA = head_outputs["cls_logits"].size(1)
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

        # split outputs per level
        split_head_outputs: Dict[str, List[Tensor]] = {}
        for k in head_outputs:
            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        # compute the detections
        detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)

        


