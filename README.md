# ShadowSense

## Unsupervised Domain Adaptation and Feature Fusion for Shadow-Agnostic Tree Crown Detection from RGB-Thermal Drone Imagery

This is the official code repository for the paper titled above, accepted into the main conference at WACV 2024! 

Two ways to access the paper:
- [CVF open access](https://openaccess.thecvf.com/content/WACV2024/html/Kapil_ShadowSense_Unsupervised_Domain_Adaptation_and_Feature_Fusion_for_Shadow-Agnostic_Tree_WACV_2024_paper.html)
- [ArXiv pre-print version](https://arxiv.org/abs/2310.16212)




<p align="center">
   <img src="images/challenge.png" alt="Training workflow" width="55%"/>
</p>


**Abstract:** Accurate detection of individual tree crowns from remote sensing data poses a significant challenge due to the dense nature of forest canopy and the presence of diverse environmental variations, e.g., overlapping canopies, occlusions, and varying lighting conditions. Additionally, the lack of data for training robust models adds another limitation in effectively studying complex forest conditions. This paper presents a novel method for detecting shadowed tree crowns and provides a challenging dataset comprising roughly 50k paired RGB-thermal images to facilitate future research for illumination-invariant detection. The proposed method (ShadowSense) is entirely self-supervised, leveraging domain adversarial training without source domain annotations for feature extraction and foreground feature alignment for feature pyramid networks to adapt domain-invariant representations by focusing on visible foreground regions, respectively. It then fuses complementary information of both modalities to effectively improve upon the predictions of an RGB-trained detector and boost the overall accuracy. Extensive experiments demonstrate the superiority of the proposed method over both the baseline RGB-trained detector and state-of-the-art techniques that rely on unsupervised domain adaptation or early image fusion.



### Running on your own machine
1. Clone and enter repository
   ```Shell
   git clone https://github.com/rudrakshkapil/ShadowSense
   cd ShadowSense
   ```
   
2. Create a conda environment for this repo (Python 3.8.13 recommended version)
   ```Shell
   conda create --name shadowsense python=3.8.13
   ```
   
3. Install PyTorch from [here](https://pytorch.org/get-started/locally/). 

4. Install other required packages with pip
   ```Shell
   pip install requirements.txt
   ```



### RT-Trees Dataset

#### Download
The dataset can be download from these two links – [RGB+Eval](doi.org/10.5281/zenodo.14007908), [Thermal](https://doi.org/10.5281/zenodo.14008187). Training, validation, and testing images for the Individual Tree Crown Detection task with RGB-thermal multi-modal drone imagery are provided. This dataset is primarily meant for self-supervised training, with ground truth annotations available for only the validation and testing sets - bounding boxes for tree crowns with **difficult** boxes marked as such in the xml file.  

14 drone flights with a DJI H20T sensor were conducted from July to November 2022. In each, ~800 image pairs (RGB & thermal) were captured in total, and the images were geographically divided into 75:5:20 train:val:test sets using GPS. The original image pairs have been cropped to 2000x2000 px. Overlap was retained in training images but removed in the evaluation sets, hence the apparent size disparity. More information regarding RT-Trees can be found in the [supplementary material](https://rudrakshkapil.com/resources/publications/wacv_supplementary.pdf) of the paper. 

| Split      | # Image Pairs | Size    |
| :---        |    :----:   |          :---: |
| Train      | 49806       | 70 GB   |
| Validation      | 10       | 16 MB   |
| Testing      | 63       | 250 MB   |


Note:
   1. Four zip files are provided, one for each of train/validation/test, and the fourth for a labellled subset of the training images, useful for comparison with supervised fine-tuning experiments. 

   2. Each split of the dataset also contains pre-computed masks for every RGB-thermal pair, obtained using watershed segmentation & morpohological operations as described in the paper and implemented in `threshold.py`.




#### Organization
The training script for the proposed self-supervised approach expects the data to be formatted in the following way
```
ShadowSense
|––– data
   |––– train
      +––– rgb
      +––– thermal
      +––– masks
   |––– test
      +––– rgb
      +––– thermal
      +––– masks
      +––– gt_annotations
   |––– val
      +––– rgb
      +––– thermal
      +––– masks
      +––– gt_annotations
```


### Training
The training flow is programmed in `train.py`. Begin training with the following command.
```Python
python train.py
```
<p align="center">
   <img src="images/workflow.png" alt="Training workflow" width="80%"/>
</p>


The evaluation metrics used are implemented in this great repo: [Open-Source Visual Interface for Object Detection Metrics](https://github.com/rafaelpadilla/review_object_detection_metrics).


### Citation
If you found this work useful for your own research, please consider citing the paper.
```bibtex
@InProceedings{Kapil_2024_WACV,
    author    = {Kapil, Rudraksh and Marvasti-Zadeh, Seyed Mojtaba and Erbilgin, Nadir and Ray, Nilanjan},
    title     = {ShadowSense: Unsupervised Domain Adaptation and Feature Fusion for Shadow-Agnostic Tree Crown Detection From RGB-Thermal Drone Imagery},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {8266-8276}
}
```
