# ShadowSense: Unsupervised Domain Adaptation and Feature Fusion for Shadow-Agnostic Tree Crown Detection from RGB-Thermal Drone Imagery
\[Under construction...\]



This is the official code repository for the paper titled above, accepted into the main conference at WACV 2024! It will be available on CVF following the conference on Jan 4-8, 2024, but for now, please refer to the [ArXiv pre-print version](https://arxiv.org/abs/2310.16212). 






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
The dataset can be download from [this Google Drive link](https://drive.google.com/drive/folders/1cCeA7TPA7qsII1-xOxs19sXTkRMV0fsl?usp=drive_link). Training, validation, and testing images for the Individual Tree Crown Detection task with RGB-thermal multi-modal drone imagery are provided. This dataset is primarily meant for self-supervised training, with ground truth annotations available for only the validation and testing sets. 

14 drone flights with a DJI H20T sensor were conducted from July to November 2022. In each, ~800 image pairs (RGB & thermal) were captured in total, and the images were geographically divided into train/val/test sets using GPS. The original image pairs have been cropped to 2000x2000 px. Overlap was retained in training images but removed in the evaluation sets, hence the apparent size disparity.


| Split      | # Image Pairs | Size    |
| :---        |    :----:   |          :---: |
| Train      | 49806       | 70 GB   |
| Validation      | 10       | 16 MB   |
| Testing      | 63       | 250 MB   |


Note:
   1. Four zip files are provided, one for each of train/validation/test, and the fourth for a labellled subset of the training images, useful for comparison with supervised fine-tuning experiments. 

   2. Each split of the dataset also contains pre-computed masks for every RGB-thermal pair, obtained using watershed segmentation & morpohological operations as described in the paper and implemented in `threshold,.py`.

More information regarding RT-Trees can be found in the [supplementary material](https://rudrakshkapil.com/resources/publications/wacv_supplementary.pdf) of the paper. 



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

The evaluation metrics used are implemented in []().


### Citation
If you found this work useful, please consider citing the arxiv paper – [arXiv:2310.16212](https://arxiv.org/abs/2310.16212) :)
Once the CVF version is available, the bibtex will be included here. 

