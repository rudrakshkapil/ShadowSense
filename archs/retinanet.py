import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

import torch
from torch import nn, Tensor
from skimage.transform import resize

from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.retinanet import RetinaNetHead, _default_anchorgen, AnchorGenerator, RetinaNet, RetinaNetRegressionHead, RetinaNetClassificationHead, _sum, _box_loss, _v1_to_v2_weights, misc_nn_ops, sigmoid_focal_loss
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.utils import _log_api_usage_once
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.image_list import ImageList


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

        if mode == 'rgb':
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
            # image_mean = [0.5,0.5,0.5]
            # image_std = [0,0,0]
        elif mode == 'thermal':
            image_mean = [13.7140]
            image_std = [1.8340]
        
            # image_mean = [0.1897]
            # image_std = [0.1112]
            # image_mean = [1]
            # image_std = [0]
        
        score_thresh = 0.1 
        nms_thresh = 0.15 # 0.15 best

        self.distill = False
        head = RetinaNetHead_Distill(backbone.out_channels, 9, 1) if self.distill else None

        super().__init__(backbone, num_classes, 
                         image_mean=image_mean, image_std=image_std, 
                         score_thresh=score_thresh, nms_thresh=nms_thresh,
                         min_size=500,max_size=500, head=head)

        # TODO: maybe head features. also, maybe anchor gen as well for more anchors (not feats), but that may lead to weights mismatch
        # self.head = UTDARetinaNetHead(backbone.out_channels, self.anchor_generator.num_anchors_per_location()[0], num_classes)

        

    def forward(self, images):
        '''
        Input: transformed images (tensors)
        '''
        # if type(images) is Tensor:
        #     images, _ = self.transform(images)

        images, _ = self.transform(images)
    
        # get the features from the backbone
        features_res, features_fpn = self.backbone(images.tensors)
        if isinstance(features_fpn, torch.Tensor):
            features_fpn = OrderedDict([("0", features_fpn)])
        if isinstance(features_res, torch.Tensor):
            features_res = OrderedDict([("0", features_res)])

 
        features_fpn = list(features_fpn.values())
        features_res = list(features_res.values())

        # compute the retinanet heads outputs using the features 
        # TODO: maybe also use head_features? 
        head_outputs = self.head(features_fpn)

        # return features (and head outputs)
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

        


    

class TDARetinanetFull(nn.Module):
    """
    Triple Domain Adaptation RetinaNet architecture.
    Allows having DA modules at ResNet layers, FPN layers, and head layers
    Same as PyTorch Retinanet, but with different methods:
     - init()
     - forward()
     - compute_losses()

    Note: backbone input is also different - here we pass in separate ResNet and FPN models instead
    of combined 'backbone'
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
    }

    def __init__(
        self,
        resnet,
        FPN,
        num_classes,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # Anchor parameters
        anchor_generator=None,
        head=None,
        proposal_matcher=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        topk_candidates=1000,
        **kwargs,
    ):
        super().__init__()
        _log_api_usage_once(self)

        self.resnet = resnet
        self.FPN = FPN

        if anchor_generator is None:
            anchor_generator = _default_anchorgen()
        self.anchor_generator = anchor_generator

        if head is None:
            head = RetinaNetHead(FPN.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
        self.head = head

        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.proposal_matcher = proposal_matcher

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # TODO: find mean and std of tree data
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(isinstance(boxes, torch.Tensor), "Expected target boxes to be of type Tensor.")
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        "Expected target boxes to be a tensor of shape [N, 4].",
                    )

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        # get the features from the backbone
        resnet_features = self.resnet(images.tensors)
        features = self.FPN(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                # compute the losses
                losses = self.compute_loss(targets, head_outputs, anchors)
        else:
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



## HEADS:


class RetinaNetHead_Distill(RetinaNetHead):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    def __init__(self, in_channels, num_anchors, num_classes, norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__(in_channels, num_anchors, num_classes)
        self.classification_head = RetinaNetClassificationHead_Distill(
            in_channels, num_anchors, num_classes, norm_layer=norm_layer
        )
        self.regression_head = RetinaNetRegressionHead_Distill(in_channels, num_anchors, norm_layer=norm_layer)


class RetinaNetClassificationHead_Distill(RetinaNetClassificationHead):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    def __init__(
        self,
        in_channels,
        num_anchors,
        num_classes,
        prior_probability=0.01,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__(in_channels,num_anchors,num_classes,prior_probability,norm_layer)

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []
        all_features = []

        for features in x:
            cls_logits = self.conv(features)         # Sequential Conv2dNormActivation (256 -> 256)
            all_features.append(cls_logits.clone())
            cls_logits = self.cls_logits(cls_logits) # conv2D ()            

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1), all_features


class RetinaNetRegressionHead_Distill(RetinaNetRegressionHead):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    def __init__(self, in_channels, num_anchors, norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__(in_channels, num_anchors, norm_layer)

    def forward(self, x):
        all_bbox_regression = []
        all_features = []

        for features in x:
            bbox_regression = self.conv(features)
            all_features.append(bbox_regression.clone())
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1), all_features

