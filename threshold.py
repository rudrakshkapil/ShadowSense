"""
Script implementating the described Binary Mask Generation workflow
Based on Watershed segmentation
"""

from skimage.segmentation import watershed
import numpy as np
from skimage.filters import sobel
from matplotlib import cm
from skimage import morphology 
from skimage.color import rgb2gray

def get_mask(image:np.ndarray, plot=False):
    '''
    based on ideas from: http://tonysyu.github.io/scikit-image/user_guide/tutorial_segmentation.html
    '''
    # convert to single channel grayscale if needed
    if len(image.shape) < 3:
        image = rgb2gray(image)

    # get elevation map
    elevation_map = sobel(image)

    # initialize two sets of markers (FG/BG)
    markers = np.zeros_like(image)
    markers[image < 20.0/255.0] = 1  
    markers[image > 100.0/255.0] = 2

    # apply watershed segmentation
    binary = watershed(elevation_map, markers) 

    # morphological operations (Opening, )
    binary = morphology.binary_opening(binary-1, np.ones((3,3)))  # -1 to change (1,2) -> (0,1)
    binary = morphology.binary_closing(binary, np.ones((3,3)))
    binary = morphology.binary_dilation(binary, np.ones((3,3)))

    # return 
    return binary
