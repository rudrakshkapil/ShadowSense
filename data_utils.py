""" 
Script implementing various utility functions for data preprocessing and handling.
Should not be needed, since the final output of all of these is RT-Trees, already provided
But you may find it a useful reference for preprocessing your own data and 
for descriptive stats of RT-Trees. 

NOTE: some of these functions assume exiftool is installed on your machine (in your PATH variable)
"""

from utm_converter import utm
import os
import shutil
import pandas
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import skimage.io
from skimage.util import img_as_ubyte

from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET

from df_repo.deepforest import main
from threshold import get_mask

def split_based_on_GPS(date, cutoff = (607840, 5907125)):
    '''
    Separate images into training and test sets, based on cutoff GPS corrdinates (determined manually from qgis for 08_30)
    Returns two lists of image names
    '''
    # load image names
    gps_dir = f"D:/Cynthia_Data/{date}/rgb/images"
    img_names = [fname for fname in os.listdir(gps_dir) if fname.endswith('.JPG') or fname.endswith('.jpeg')]

    # lists for storing
    train_set = []
    test_set = []

    # NOTE: need exiftool installed
    try:
        if '08_30' not in date:
            call = f"exiftool -a {gps_dir} -ext jpeg > exif_tmp.txt"
        else:
            call = f"exiftool -a {gps_dir} -ext jpg > exif_tmp.txt"
        os.system(call)
    except Exception as e:
        raise Exception("Exiftool not found, so can't split data based on GPS location!")

    def divide_chunks(l, n):
        for i in range(0, len(l), n): 
            yield l[i+1:i + n]

    with open(f"exif_tmp.txt", 'r') as f:
        lines = f.readlines()[:-2]
        len_ = len(lines)
        chunked_lines = list(divide_chunks(lines, 117))


    i = 0
    for lines in (pbar:=tqdm(chunked_lines, leave=False)):
        pbar.set_description(date)
        exif_dict = {}
        for line in lines:
            try:
                line = line.split(':')
                key = line[0].strip()
                val = line[1].strip()
                exif_dict[key] = val
            except:
                print(line)
                
        lat = exif_dict["GPS Latitude"].split(" ")
        lon = exif_dict["GPS Longitude"].split(" ")

        lat_dec = float(lat[0]) + float(lat[2][:-1])/60 + float(lat[3][:-2])/3600
        lon_dec = float(lon[0]) + float(lon[2][:-1])/60 + float(lon[3][:-2])/3600

        # convert lat,long to UTM (Lines 5 & 6)
        easting, northing, _, _ = utm.from_latlon(lat_dec, lon_dec)

        name = i
        i += 1
        if northing > cutoff[1] and easting < cutoff[0]:
            test_set.append(name)
        else: train_set.append(name)

    return train_set, test_set


def move_images(date, train_set, test_set):
    '''
    Move images from D:/Cynthia_Data to ./data
    Does it separately for splits and modes
    '''

    # get source directories for given date
    rgb_src_dir = f"D:/Cynthia_Data/{date}/combined/opensfm/undistorted/images_rgb"
    thm_src_dir = f"D:/Cynthia_Data/{date}/combined/opensfm/undistorted/images"     

    # define and make output dirs if not exists
    rgb_train_opd = f'./data/train/all/{date}/rgb/'
    rgb_test_opd  = f'./data/test/all/rgb_geo'
    thm_train_opd = f'./data/train/all/{date}/thermal/'
    thm_test_opd  = f'./data/test/all/thermal_geo'
    try:
        os.makedirs(rgb_train_opd), os.makedirs(thm_train_opd) 
        if date == '2022_08_30':
            os.makedirs(rgb_test_opd), os.makedirs(thm_test_opd)
    except: pass

    # move training set images
    for name in tqdm(train_set):
        rgb_src_path = f'{rgb_src_dir}/{name}.tif'
        thm_src_path = f'{thm_src_dir}/{name}.tif'

        shutil.copy(rgb_src_path, rgb_train_opd)
        shutil.copy(thm_src_path, thm_train_opd)

        # add exif information if exiftool present
        try:
            call = f"exiftool -tagsfromfile {rgb_src_path} \"-exif:all>exif:all\" {rgb_test_opd}/{name} >> exif_log.txt 2>&1" # pipe stderr (2) andstdout (1) to output file
            os.system(call)
            call = f"exiftool -tagsfromfile {thm_src_path} \"-exif:all>exif:all\" {thm_test_opd}/{name} >> exif_log.txt 2>&1" # pipe stderr (2) andstdout (1) to output file
            os.system(call)
        except:
            print("[Optional] Exiftool not found, skipping copying of exif info during move_images()")

    # for test set, only move 2022_08_30 images
    if date == '2022_08_30':
        for name in tqdm(test_set):
            rgb_src_path = f'{rgb_src_dir}/{name}.tif'
            thm_src_path = f'{thm_src_dir}/{name}.tif'

            shutil.copy(rgb_src_path, rgb_test_opd)
            shutil.copy(thm_src_path, thm_test_opd)

            # add exif information if exiftool present
            try:
                call = f"exiftool -tagsfromfile {rgb_src_path} \"-exif:all>exif:all\" {rgb_test_opd}/{name} >> exif_log.txt 2>&1" # pipe stderr (2) andstdout (1) to output file
                os.system(call)
                call = f"exiftool -tagsfromfile {thm_src_path} \"-exif:all>exif:all\" {thm_test_opd}/{name} >> exif_log.txt 2>&1" # pipe stderr (2) andstdout (1) to output file
                os.system(call)
            except:
                print("[Optional] Exiftool not found, skipping copying of exif info during move_images()")


def crop_single_test_image(src_path, dst_path, p=500):
    img = skimage.io.imread(src_path)
    h,w = img.shape[:2]

    patch = img[h//2-p//2:h//2+p//2, w//2-p//2:+w//2+p//2]
    skimage.io.imsave(dst_path, patch)


def crop_test_data(rgb_dir = f'data/test/all/rgb', rgb_outdir = f'data/test/rgb', do_thermal=True, p=500):
    """ Center crop test data """
    fnames = os.listdir(rgb_dir)
    paths = [f'{rgb_dir}/{fname}' for fname in fnames]
    
    try: os.makedirs(rgb_outdir)
    except: pass

    # sample every third image, crop its center, and save to rgb_outdir
    x = 3
    for idx,path in enumerate(paths[0:len(fnames):x]):
        img = skimage.io.imread(path)
        h,w,_ = img.shape

        patch = img[h//2-p//2:h//2+p//2, w//2-p//2:+w//2+p//2]
        skimage.io.imsave(f'{rgb_outdir}/{fnames[idx*3]}', patch)

    # do the same for thermal if needed
    if do_thermal:
        thm_dir = f'data/test/all/thermal'
        fnames = os.listdir(thm_dir)
        paths = [f'{thm_dir}/{fname}' for fname in fnames]

        thm_outdir = f'data/test/thermal'
        try: os.makedirs(thm_outdir)
        except: pass
        
        for idx,path in enumerate(paths[0:len(fnames):x]):
            img = skimage.io.imread(path)
            h,w = img.shape

            patch = img[h//2-p//2:h//2+p//2, w//2-p//2:+w//2+p//2]
            skimage.io.imsave(f'{thm_outdir}/{fnames[idx*3]}', patch)

    

def split_image(img, patch_size):
    ''' 
    https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7 
    Takes image and returns array of tiles, each having patch_size dimensions. 
    Output is size (# of tiles, patch_h, patch_w, 3)
    '''
    if len(img.shape) == 3:
        img_h,img_w,c = img.shape
        patch_h,patch_w = patch_size, patch_size

        tiled_array = img.reshape(img_h // patch_h, patch_h, img_w//patch_w, patch_w, c)
        tiled_array = tiled_array.swapaxes(1,2)
        tiled_array = tiled_array.reshape(-1, patch_h, patch_w, c)

    else:
        img_h,img_w = img.shape
        patch_h,patch_w = patch_size, patch_size

        tiled_array = img.reshape(img_h // patch_h, patch_h, img_w//patch_w, patch_w)
        tiled_array = tiled_array.swapaxes(1,2)
        tiled_array = tiled_array.reshape(-1, patch_h, patch_w)

    return tiled_array



def patchify_train_data(date, p=500):
    """
    Split train images into 6x500x500 patches
    """
    # get rgb image paths and make rgb out directory
    rgb_dir = f'data/train/all/{date}/rgb'
    rgb_fnames = os.listdir(rgb_dir)
    rgb_paths = [f'{rgb_dir}/{fname}' for fname in rgb_fnames]

    rgb_outdir = f'data/train/rgb'
    try: os.makedirs(rgb_outdir)
    except: pass

    # get rgb image paths and make rgb out directory
    thm_dir = f'data/train/all/{date}/thermal'
    thm_fnames = os.listdir(thm_dir)
    thm_paths = [f'{thm_dir}/{fname}' for fname in thm_fnames]

    thm_outdir = f'data/train/thermal'
    try: os.makedirs(thm_outdir)
    except: pass

    mode = " RGB"

    def move_images_from_dates(paths, fnames, out_dir):
        for idx,path in enumerate((pbar:=tqdm(paths, leave=False))):
            # plt.subplot(2,5,idx+1)
            pbar.set_description(date+mode)
            img = skimage.io.imread(path)
            h,w = img.shape[:2]

            # crop within bounds s.t. we have int number of patches
            rem_w = w % p
            start_w, end_w = rem_w//2, w - rem_w//2
            rem_h = h % p
            start_h, end_h = rem_h//2, h - rem_h//2

            # make patches of cropped image
            cropped =  img[start_h:end_h, start_w:end_w]
            patches = split_image(cropped, p)

            # paths for saving
            file_id = str(int(fnames[idx].split('.')[0].split('_')[1])+1).zfill(3)
            date_id = ''.join(date.split('_')[1:])
        
            # loop over patches and save
            for p_idx, patch in enumerate(patches):
                out_path = f"{date_id}_{file_id}_{p_idx+1}.tif"
                out_path = f"{out_dir}/{out_path}"
                p_idx+1
                skimage.io.imsave(out_path, patch)

    move_images_from_dates(rgb_paths, rgb_fnames, rgb_outdir)
    mode = " Thermal"
    move_images_from_dates(thm_paths, thm_fnames, thm_outdir)



def write_xml(path, filename, bbox_list, save_dir_suffix, size=[500,500]):
    ''' 
    Utility function to write an entire xml file 
    Input: 
        - path: 'folder_parent/folder'
        - filename: all must be in folder
    '''
    
    root = Element('annotation')
    folder = path.split('/')[-1] # last part of path
    SubElement(root, 'folder').text = folder
    SubElement(root, 'filename').text = filename
    SubElement(root, 'path').text = path
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'

    # get size
    height, width = size
    
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(width)
    SubElement(size, 'height').text = str(height)
    SubElement(size, 'depth').text = '3'
    SubElement(root, 'segmented').text = '0'

    # TODO: maybe also put score somewhere
    for entry in bbox_list:
        xmin, ymin, xmax, ymax = entry
        
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = 'Tree'
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = str(xmin)
        SubElement(bbox, 'ymin').text = str(ymin)
        SubElement(bbox, 'xmax').text = str(xmax)
        SubElement(bbox, 'ymax').text = str(ymax)

    ET.indent(root, space="\t")
    tree = ElementTree(root)
    
    save_dir = path[:-len(folder)] + save_dir_suffix
    xml_filename = os.path.join(save_dir, os.path.splitext(filename)[0] + '.xml')
    with open(xml_filename, 'w'):
        pass
    tree.write(xml_filename)


def save_gt_only_difficult(gt_dir, save_dir):
    """
    Save gt annotations for difficult boxes
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files = os.listdir(gt_dir)
    for gt_file in tqdm(files):
        tree = ET.parse(f'{gt_dir}/{gt_file}')
        root = tree.getroot()
        
        count = 0
        for idx,obj in enumerate(root[:5:-1]):
            if obj[3].text != '1':
                root.remove(obj)
                count += 1

        tree = ElementTree(root)
    
        xml_filename = os.path.join(save_dir, gt_file)
        with open(xml_filename, 'w'):
            tree.write(xml_filename)


# -------- Descriptive statistics of RT-Trees --------
def count_difficult(gt_dir):
    """
    Count number of difficult boxes in all images within passed annotations dir.
    """
    count,total = 0,0
    l1, l2 = [], []
    files = os.listdir(gt_dir)
    for gt_file in tqdm(files):
        tree = ET.parse(f'{gt_dir}/{gt_file}')
        root = tree.getroot()

        curr_count, curr_total = 0, 0
        
        for idx,obj in enumerate(root[:5:-1]):
            if obj[3].text == '1':
                curr_count +=1
            curr_total += 1
        l1.append(curr_count)
        l2.append(curr_total)
        count += curr_count
        total += curr_total
        
    print(f"Total Difficult in {len(files)} images => {count}/{total}")
    return count, total, l1, l2
       

def get_areas(gt_dir):
    """
    Compute areas of all/difficult boxes in all images within passed annotations dir.
    """
    files = os.listdir(gt_dir)
    all_areas, other_areas, difficult_areas = [], [], []
    for gt_file in tqdm(files):
        tree = ET.parse(f'{gt_dir}/{gt_file}')
        root = tree.getroot()
        objs = root[6:]
        for obj in objs:
            xmin = int(obj[4][0].text)
            ymin = int(obj[4][1].text)
            xmax = int(obj[4][2].text)
            ymax = int(obj[4][3].text)

            area = (xmax-xmin) * (ymax-ymin) /(500*500) * 100

            all_areas.append(area)
            if obj[3].text == '1':
                difficult_areas.append(area)
            else: other_areas.append(area)
    return all_areas, other_areas, difficult_areas

    
def get_HW(gt_dir):
    """
    Compute height and width of all/difficult boxes in all images within passed annotations dir.
    """
    files = os.listdir(gt_dir)
    all_areas, other_areas, difficult_areas = [], [], []
    for gt_file in tqdm(files):
        tree = ET.parse(f'{gt_dir}/{gt_file}')
        root = tree.getroot()
        objs = root[6:]
        for obj in objs:
            xmin = int(obj[4][0].text)
            ymin = int(obj[4][1].text)
            xmax = int(obj[4][2].text)
            ymax = int(obj[4][3].text)

            (h,w) = (ymax-ymin, xmax-xmin)
            # d = {'h':h, 'w':w}
            d = (h,w)

            all_areas.append(d)
            if obj[3].text == '1':
                difficult_areas.append(d)
            else: other_areas.append(d)
    return all_areas, other_areas, difficult_areas

   
def get_box_brightness(gt_dir, rgb_dir, thm_dir):
    """
    Get average LAB-space brightness of images within passed dir.
    """
    files = os.listdir(gt_dir)
    all_areas, other_areas, difficult_areas = [], [], []
    full_image = []
    for gt_file in tqdm(files):
        tree = ET.parse(f'{gt_dir}/{gt_file}')
        root = tree.getroot()
        objs = root[6:]

        rgb_img = skimage.color.rgb2lab(skimage.io.imread(f'{rgb_dir}/{gt_file.split(".")[0]}.tif'))[:,:,0]
        thm_img = skimage.io.imread(f'{rgb_dir}/{gt_file.split(".")[0]}.tif')

        def min_max_norm(x): return (x-np.amin(x)) / (np.amax(x) - np.amin(x))
        # def min_max_norm(x): return (x-np.mean(x)) / (np.std(x))
        # def min_max_norm(x): return x
        rgb_img = min_max_norm(rgb_img)
        thm_img = min_max_norm(thm_img)


        full_image.append([rgb_img.mean(), thm_img.mean()])
        for obj in objs:
            xmin = int(obj[4][0].text)
            ymin = int(obj[4][1].text)
            xmax = int(obj[4][2].text)
            ymax = int(obj[4][3].text)

            rgb_patch = rgb_img[ymin:ymax, xmin:xmax]
            thm_patch = thm_img[ymin:ymax, xmin:xmax]
            d = (rgb_patch.mean(), thm_patch.mean())

            all_areas.append(d)
            if obj[3].text == '1':
                difficult_areas.append(d)
            else: other_areas.append(d)
    return all_areas, other_areas, difficult_areas, full_image


def get_brightness_by_date():
    """ Save brightness of images by flight date to csv """
    def get_brightness(dir): 
        fnames = os.listdir(dir) 
        avg = []
        for f in tqdm(fnames[100:200]):
            img = skimage.io.imread(f'{dir}/{f}')
            tform_img = skimage.color.rgb2lab(img)
            avg.append(np.mean(tform_img[:,:,0]))
        return avg

    date_names = ['July 20', 'July 26', 'August 9', 'August 17', 'August 30', 'September 9', 'September 15', 'September 23', 'October 4', 'October 6','October 7', 'October 12', 'October 19', 'November 24']
    df = pandas.DataFrame(columns=['Brightness (units)', 'Flight Date'])

    root = "D:/Cynthia_Data"
    dates = [d for d in os.listdir("D:/Cynthia_Data") if d.startswith('2022') and d not in ('2022_08_03', '2022_06_03', '2022_07_12') and 'test' not in d]
    for idx, date in enumerate(dates[1:]):
        print(date, date_names[idx])
        l_ = get_brightness(f'{root}/{date}/combined/images_rgb')
        d_ = {'Brightness (units)': l_, 'Flight Date': date_names[idx]}
        d_ = pandas.DataFrame(d_)
        df = pandas.concat([df, d_])
    df.to_csv('plots/brightness.csv')


def get_image_pairs():
    """ Load image pairs and plot specific ones -- only used once for creating a diagram """
    root = "D:/Cynthia_Data"
    dates = [d for d in os.listdir("D:/Cynthia_Data") if d.startswith('2022') and d not in ('2022_08_03', '2022_06_03', '2022_07_12') and 'test' not in d]
    
    all_image_pairs = []

    for _, date in enumerate(dates[1:]):
        try:
            rgb_img = skimage.io.imread(f'{root}/{date}/combined\opensfm\\undistorted\images_rgb\img_00100.jpeg.tif')
            thm_img = skimage.io.imread(f'{root}/{date}/combined\opensfm\\undistorted\images\img_00100.jpeg.tif')
        except:            
            rgb_img = skimage.io.imread(f'{root}/{date}/combined\opensfm\\undistorted\images_rgb\img_00100.JPG.tif')
            thm_img = skimage.io.imread(f'{root}/{date}/combined\opensfm\\undistorted\images\img_00100.JPG.tif')

        # get central region
        rgb_img = rgb_img[108:-108,61:-61,:]
        thm_img = thm_img[108:-108,61:-61]

        # normalize
        def min_max_norm(x): return (x-np.amin(x)) / (np.amax(x) - np.amin(x))
        thm_img = min_max_norm(thm_img)

        # store, and return after loop
        all_image_pairs.append([rgb_img, thm_img])
    return all_image_pairs


def save_pseudo_labels(dir, save_dir_suff):
    """ Save Deepforest prediction on images within passed dir """
    model = main.deepforest()
    model.use_release()
    model.cuda()

    save_dir = '/'.join(dir.split('/')[:-1]) + '/' + save_dir_suff
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # get image paths 
    fnames = [fname for fname in os.listdir(dir) if fname.endswith(".tif")]

    # output file for images with no predictions (barren)
    f = open("tree_less.txt", 'w')

    # get predictions
    for fname in tqdm(fnames):
        img = skimage.io.imread(f'{dir}/{fname}').astype('float32')
        pred = model.predict_image(image=img)

        if pred is None:
            f.write(fname+'\n')
            print(fname)
            continue

        bboxes = []
        for _, row in pred.iterrows():
            bboxes.append([int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])])
        write_xml(dir, fname, bboxes, save_dir_suff)

    f.close()


def convert_thermal_to_ubyte(thm_dir, save_dir):
    """ Save thermal images (tiffs) as ubyte for plotting purposes """
    # make save dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # read filenames
    fnames = os.listdir(thm_dir)

    # read, save as ubyte
    for fname in tqdm(fnames):
        img = skimage.io.imread(f'{thm_dir}/{fname}')
        min_, max_ = np.amin(img), np.amax(img)
        img = (img-min_)/(max_-min_)
        img = img_as_ubyte(img)
        skimage.io.imsave(f'{save_dir}/{fname}', img)

        
def remove_files_with_ext(dir, ext):
    """ Utility to delete images from dir that end in .<ext> """
    fnames = os.listdir(dir)
    for fname in fnames:
        if ext in fname:
            os.remove(f'{dir}/{fname}')
    
    print(f"Removed. Remaining: {len(os.listdir(dir))}")
    

def plot_path(test_dir):
    """ Plot path of drone flight using GPS information"""
    gps_dir = f"D:/Cynthia_Data/2022_08_30/combined/images_rgb"
    test_imgs = os.listdir(test_dir)
    img_names_test = [fname for fname in os.listdir(gps_dir) if fname+'.tif' in test_imgs]
    img_names = os.listdir(gps_dir)

    xs, ys = [], []
    xs_test, ys_test = [],[]

    # read exifs (height, width, aperture size, altitude, gps)
    for name in (pbar:=tqdm(img_names, leave=False)): 
        # extract exif info to a temp file
        call = f"exiftool -a {gps_dir}/{name} > ./exif_tmp.txt"
        os.system(call)
        exif_dict = {}
        with open(f"exif_tmp.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(':')
                key = line[0].strip()
                val = line[1].strip()
                exif_dict[key] = val

        # GPS from LRF values directly
        lat = exif_dict["GPS Latitude"].split(" ")
        lon = exif_dict["GPS Longitude"].split(" ")

        lat_dec = float(lat[0]) + float(lat[2][:-1])/60 + float(lat[3][:-2])/3600
        lon_dec = -(float(lon[0]) + float(lon[2][:-1])/60 + float(lon[3][:-2])/3600)

        # convert lat,long to UTM (Lines 5 & 6)
        easting, northing, _, _ = utm.from_latlon(lat_dec, lon_dec)
        if name in img_names_test:
            xs_test.append(easting)
            ys_test.append(northing)
        else:
            xs.append(easting)
            ys.append(northing)

    # print(xs,ys)
    save_dir = './plots/data_split_points'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savetxt(save_dir+'/xs.txt', np.array(xs))
    np.savetxt(save_dir+'/ys.txt', np.array(ys))
    np.savetxt(save_dir+'/xs_test.txt', np.array(xs_test))
    np.savetxt(save_dir+'/ys_test.txt', np.array(ys_test))


    # NOTE: to save time, can skip to this loading step if need to repeat this function
    xs = list(np.loadtxt(save_dir+'/xs.txt'))
    ys = list(np.loadtxt(save_dir+'/ys.txt'))
    xs_test = list(np.loadtxt(save_dir+'/xs_test.txt'))
    ys_test = list(np.loadtxt(save_dir+'/ys_test.txt'))

    xs_val, ys_val = [], []
    for i in xs[:30]:
        xs_val.append(i)
        xs.remove(i)
    for i in ys[:30]:
        ys_val.append(i)
        ys.remove(i)

    # plot
    plt.plot(xs, ys, '.', color = 'orange', label=f'Training')
    plt.plot(xs_test[::3], ys_test[::3], '+', color='teal', label=f'Testing')
    plt.plot(xs_val[::3], ys_val[::3], 'x', color="purple", label='Validation')
    plt.axhline(y=5907125, color='black', linestyle='--')
    plt.text(608050, 5907105, 'cutoff: 5901725 m', style='italic')
    plt.xlabel('Easting (meters)')
    plt.ylabel('Northing (meters)')
    plt.title('GPS-based Data Split for August 30 Flight')
    plt.legend()
    plt.savefig('./plots/data_split.pdf', bbox_inches='tight')


def save_binary_masks(img_dir, save_dir):
    """ Save binary masks of images (using function in ./threshold.py) into save_dir """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fnames = os.listdir(img_dir)
    for fname in tqdm(fnames):
        path = f'{img_dir}/{fname}'
        image = skimage.io.imread(path, as_gray=True)
        binary = torch.tensor(get_mask(image))
        torch.save(binary, f'{save_dir}/{fname}.pt')


def save_binary_masks_boxes(ann_dir, save_dir, size=(500,500)):
    """ Save binary masks of images (using Deepforest-predicted boxes as FG, rest as BG) into save_dir """
    # 2. load pseudo boxes and construct mask
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fnames = os.listdir(ann_dir)

    for gt_file in tqdm(fnames):
        tree = ET.parse(f'{ann_dir}/{gt_file}')
        root = tree.getroot()
        objs = root[6:]
        mask = torch.zeros(size, dtype=torch.bool)
        for obj in objs:
            xmin = int(obj[4][0].text)
            ymin = int(obj[4][1].text)
            xmax = int(obj[4][2].text)
            ymax = int(obj[4][3].text)
            mask[ymin:ymax, xmin:xmax] = True
        torch.save(mask, f'{save_dir}/{gt_file}.pt')


def save_weighted_boxes(dir, save_dir_suff):
    """ 
    Save binary masks of images (using Deepforest-predicted boxes as FG, rest as BG) into save_dir 
    This time, the masks are weighted in a Gaussian manner with the mean at the center. 
    """
    # deepforest model
    model = main.deepforest()
    model.use_release()
    model.cuda()

    save_dir = '/'.join(dir.split('/')[:-1]) + '/' + save_dir_suff
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # get image paths 
    fnames = [fname for fname in os.listdir(dir) if fname.endswith(".tif")]
    # fnames = ['img_14.tif', 'img_15.tif', 'img_55.tif']

    f = open("tree_less.txt", 'w')

    # get predictions
    for fname in tqdm(fnames):
        img = skimage.io.imread(f'{dir}/{fname}').astype('float32')
        pred = model.predict_image(image=img)

        if pred is None:
            f.write(fname+'\n')
            print(fname)
            continue

        mask = torch.zeros(img.shape[:2], dtype=torch.float32)
        for _, row in pred.loc[::-1].iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            score = row['score']
            mask[ymin:ymax, xmin:xmax] = score

        torch.save(mask, f'{save_dir}/{fname.split(".")[0]}.xml.pt')

    f.close()


def copy_every_nth_image_and_labels(n = 24):
    """ Make a copy of every nth image in the source directories to the target directories (hardcoded) """
    img_dir = 'data/train/rgb'
    lbl_dir = 'data/train/pseudo_annotations'
    new_img_dir = 'data/train/rgb_sub'
    new_lbl_dir = 'data/train/gt_annotations_sub'
    thm_dir = 'data/train/thermal'
    new_thm_dir = 'data/train/thermal_sub'

    try:
        os.makedirs(new_img_dir)
        os.makedirs(new_thm_dir)
        os.makedirs(new_lbl_dir)
    except: pass

    img_names = os.listdir(img_dir)
    lbl_names = os.listdir(lbl_dir)

    for i in tqdm(range(0,len(img_names),n)):
        shutil.copyfile(f'{img_dir}/{img_names[i]}', f'{new_img_dir}/{img_names[i]}')
        shutil.copyfile(f'{thm_dir}/{img_names[i]}', f'{new_thm_dir}/{img_names[i]}')
        shutil.copyfile(f'{lbl_dir}/{lbl_names[i]}', f'{new_lbl_dir}/{lbl_names[i]}')


def add_missing():
    """ Add missing images in target directories from source directories (hardcoded) """
    lbl_dir = 'data/train/pseudo_annotations'
    new_img_dir = 'data/train/rgb_sub'
    new_lbl_dir = 'data/train/gt_annotations_sub'
    thm_dir = 'data/train/thermal'
    new_thm_dir = 'data/train/thermal_sub'

    img_names = os.listdir(new_img_dir)

    for name in tqdm(img_names):
        src = f'{lbl_dir}/{name[:-4]}.xml'
        dst = f'{new_lbl_dir}//{name[:-4]}.xml'
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copyfile(src, dst)


def convert_gt_to_DF_compatible(gt_dir, out_path=None):
    """ Convert GT annotations to a format compatible with Deepforest """
    # get fnames
    fnames = os.listdir(gt_dir)

    # output path
    if out_path is None:
        out_path = 'data/subset/annotations.csv'

    with open(out_path, 'w') as f:
        for gt_file in tqdm(fnames):
            tree = ET.parse(f'{gt_dir}/{gt_file}')
            root = tree.getroot()
            objs = root[6:]
            for obj in objs:
                xmin = int(obj[4][0].text)
                ymin = int(obj[4][1].text)
                xmax = int(obj[4][2].text)
                ymax = int(obj[4][3].text)

                name = gt_file.split('.')[0]+'.png'
                if xmax > xmin and ymax > ymin:
                    f.write(f"{name}, {xmin}, {ymin}, {xmax}, {ymax},Tree\n")
    
    df = pandas.read_csv(out_path)
    df.columns = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    df.to_csv(out_path, index=False)
    



