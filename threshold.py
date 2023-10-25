from skimage.filters import threshold_otsu
import skimage.io
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
import numpy as np
from skimage.filters import sobel
from matplotlib import cm
from skimage import morphology 

from scipy import ndimage as ndi

def get_mask(image:np.ndarray):
    '''
    based on ideas from: http://tonysyu.github.io/scikit-image/user_guide/tutorial_segmentation.html
    '''

    elevation_map = sobel(image)
    print(elevation_map.shape)
    xs = np.arange(1,501)
    ys = np.arange(1,501)
    xs, ys = np.meshgrid(xs, ys)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xs,ys,elevation_map, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_title('Elevation Map')
    plt.show()

    markers = np.zeros_like(image)
    markers[image < 20.0/255.0] = 1   # NOTE: optimal
    markers[image > 100.0/255.0] = 2
    # markers[image < 10.0/255.0] = 1
    # markers[image > 125.0/255.0] = 2

    binary = watershed(elevation_map, markers) 
    # plt.figure(figsize=(14,4))
    plt.subplot(141), plt.imshow(binary, cmap='gray'), plt.title('Watershed')

    # binary = ndi.binary_fill_holes(binary - 1)
    # binary = morphology.binary_opening(binary-1, np.ones((2,2)))
    # plt.subplot(143), plt.imshow(binary), plt.title('Area Opening')
    # binary = morphology.binary_closing(binary, np.ones((2,2)))
    # plt.subplot(142), plt.imshow(binary), plt.title('Hole Filling')
    # binary = morphology.binary_dilation(binary, np.ones((2,2)))
    # plt.subplot(144), plt.imshow(binary), plt.title('Dilation')
    # plt.subplot(142), plt.imshow(binary), plt.title('Hole Filling')
    # 
    # binary = ndi.grey_closing(binary, (3,3))
    # binary = ndi.grey_opening(binary, (4,4))
    binary = morphology.binary_opening(binary-1, np.ones((3,3)))        # opening (2x2), closing (2x2), dilation best so far (55.36)
    plt.subplot(142), plt.imshow(binary, cmap='gray'), plt.title('Opening')
    binary = morphology.binary_closing(binary, np.ones((3,3)))
    plt.subplot(143), plt.imshow(binary, cmap='gray'), plt.title('Closing')

    # binary = ndi.binary_fill_holes(binary)
    # plt.subplot(143), plt.imshow(binary), plt.title('Area Opening')
    # binary = ndi.grey_dilation(binary, (2,2))
    binary = morphology.binary_dilation(binary, np.ones((3,3)))
    plt.subplot(144), plt.imshow(binary, cmap='gray'), plt.title('Dilation')

    # binary = ndi.binary_fill_holes(binary)
    return binary

def thresh_trial():
    path = 'data/test/rgb/img_08.tif'
    image = skimage.io.imread(path, as_gray=True)

    # thresh = threshold_otsu(image)
    # binary = image > thresh
    # image = binary.astype(np.uint8)

    # # hole filling
    # image = ndi.binary_fill_holes(image).astype(np.uint8)
    # # binary = ndi.binary_fill_holes(binary)
    # binary[binary == 0] = 2


    binary = get_mask(image)
    # 

    # min_, max_ = np.amin(image), np.amax(image)
    # print(min_, max_)
    # # image = ((image-min_)/(max_-min_)*255.0).astype(np.uint8)

    # elevation_map = sobel(image)
    # print(elevation_map.shape)
    # xs = np.arange(1,501)
    # ys = np.arange(1,501)
    # xs, ys = np.meshgrid(xs, ys)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(xs,ys,elevation_map, cmap=cm.coolwarm,
    #                    linewidth=0, antialiased=False)
    # ax.set_title('Elevation Map')
    # plt.show()

    # # markers = np.zeros_like(image)
    # # markers[coin < ]

    # markers = np.zeros_like(image)
    # markers[image < 20.0/255.0] = 1
    # markers[image > 100.0/255.0] = 2

    # binary = watershed(elevation_map, markers) 
    # # markers[image < 7.0/np.amax(image)] = 1
    # # markers[image > 17.0/np.amax(image)] = 2

    # plt.figure(figsize=(14,4))
    # plt.subplot(141), plt.imshow(binary), plt.title('Watershed')

    
    # # plt.subplot(142), plt.imshow(binary), plt.title('Hole Filling')
    # # 
    # # binary = ndi.grey_closing(binary, (3,3))
    # # binary = ndi.grey_opening(binary, (4,4))
    
    # # binary = ndi.binary_fill_holes(binary)
    # # plt.subplot(143), plt.imshow(binary), plt.title('Area Opening')
    # # binary = ndi.grey_dilation(binary, (2,2))
    # # binary = morphology.dilation(binary)
    # # binary = ndi.binary_fill_holes(binary)
    


    
    # binary = morphology.binary_opening(binary-1, np.ones((5,5)))
    # # binary = morphology.binary_opening(binary, np.ones((5,5)))
    # # binary = morphology.binary_opening(binary, np.ones((4,4)))
    # plt.subplot(143), plt.imshow(binary), plt.title('Area Opening')
    # binary = morphology.binary_closing(binary, np.ones((2,2)))
    # plt.subplot(142), plt.imshow(binary), plt.title('Hole Filling')
    # binary = morphology.binary_dilation(binary, np.ones((2,2)))
    # plt.subplot(144), plt.imshow(binary), plt.title('Dilation')
    # plt.show() 


    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

    ax[1].hist(image.ravel(), bins=256)
    ax[1].set_title('Histogram')
    # ax[1].axvline(thresh, color='r')

    image = skimage.io.imread(path)
    ax[0].imshow(image)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[2].imshow(binary, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded')
    ax[2].axis('off')

    plt.show()


if __name__ == "__main__":
    thresh_trial()
