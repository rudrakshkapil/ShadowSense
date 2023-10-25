
import os
import torch
import numpy as np
import pandas as pd
import skimage
import skimage.color
import skimage.io
from skimage.transform import resize
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch.utils.data import DataLoader
from tqdm import tqdm
from archs.retinanet import UTDARetinanet
from dataloaders import RGBxThermalTreeCrownDataset

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from archs.RGBTdetector import RGBxThermalDetector
from archs.encDec import UNet
from threshold import get_mask

from df_repo.deepforest import main as df_main




# TODO: t-sne
def tsne_vis(features, labels, palette=['blue','red'], title="T-SNE plot", perplexity=30, pca=False):
    # pca
    if pca:
        print("Performing PCA....")
        pca_50 = PCA(n_components=50, random_state=0)
        features = pca_50.fit_transform(features)
        print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))


    # perform t-sne
    tsne = TSNE(n_components=2, verbose=1, random_state=0, perplexity=perplexity, n_iter=1000)
    z = tsne.fit_transform(features)

    # extract data 
    df = pd.DataFrame()
    df["y"] = labels
    df["Dimension 1"] = z[:,0]
    df["Dimension 2"] = z[:,1]

    # plot and save figure
    g1 = sns.scatterplot(x="Dimension 1", y="Dimension 2", hue=df.y.tolist(),
                palette=palette,
                data=df, legend=False).set(title=title) 


def visualize_res_features(model, eval_dataloader):
    model.cpu()
    all_rgb_feats = {'res3':None, 'res4':None, 'res5':None}
    all_thm_feats = {'res3':None, 'res4':None, 'res5':None}

    for batch in tqdm(eval_dataloader):
        # get input on gpu
        x_rgb, x_thermal, _ = batch
        # x_rgb, x_thermal = x_rgb.cuda(), x_thermal.cuda()

        # extract features
        rgb_feats_res, _, _ = model.rgb_detector(x_rgb)
        thm_feats_res, _, _ = model.thermal_detector(model.thermal_prelayer(x_thermal))

        def append_list_to_dict(d, l):
            # append
            if d['res3'] is None:
                d['res3'] = l[0]
                d['res4'] = l[1]
                d['res5'] = l[2]
            else:
                d['res3'] = torch.cat([d['res3'], l[0]])
                d['res4'] = torch.cat([d['res4'], l[1]])
                d['res5'] = torch.cat([d['res5'], l[2]])
            
        append_list_to_dict(all_rgb_feats, rgb_feats_res)
        append_list_to_dict(all_thm_feats, thm_feats_res)

    # concatenate and make labels list (0: rgb source, 1: thermal target)
    n_images = len(eval_dataloader.dataset)
    all_feats = {}
    for k, v in all_rgb_feats.items():
        all_feats[k] = torch.cat([all_rgb_feats[k], all_thm_feats[k]])
    labels = ['RGB']*n_images + ['Thermal']*n_images

    # get features as flattened numpy arrays
    def to_numpy(dict_):
        for key,val in dict_.items():
            val = val.to('cpu').detach().numpy()
            val = np.reshape(val, (n_images*2, -1))
            dict_[key] = val
    to_numpy(all_feats)


    plt.figure(figsize=(15,5))
    plt.subplot(131)
    tsne_vis(features=all_feats['res3'], labels=labels, title='Res3')
    plt.subplot(132)
    tsne_vis(features=all_feats['res4'], labels=labels, title='Res4')
    plt.subplot(133)
    tsne_vis(features=all_feats['res5'], labels=labels, title='Res5')
    plt.show()
    model.cuda()

def visualize_fpn_features(model, eval_dataloader, start):
    all_rgb_feats = None
    all_thm_feats = None
    model.cpu()

    for batch in tqdm(eval_dataloader):
        # get input on gpu
        x_rgb, x_thermal, _ = batch
        # x_rgb, x_thermal = x_rgb.cuda(), x_thermal.cuda()

        # extract featurs
        _, rgb_feats_fpn, _ = model.rgb_detector(x_rgb)
        _, thm_feats_fpn, _ = model.thermal_detector(model.thermal_prelayer(x_thermal))
        

        def append_list_to_biglist(bigl, l):
            # append
            if bigl is None:
                bigl = l
            else:
                for i in range(5):
                    bigl[i] = torch.cat([bigl[i], l[i]])
            return bigl
            
        all_rgb_feats = append_list_to_biglist(all_rgb_feats, rgb_feats_fpn)
        all_thm_feats = append_list_to_biglist(all_thm_feats, thm_feats_fpn)

    print([l.shape for l in all_rgb_feats])

    # concatenate and make labels list (0: rgb source, 1: thermal target)
    n_images = len(eval_dataloader.dataset)
    all_feats = [None]*n_images
    for i in range(5):
        # flatten and get np array
        val = torch.cat([all_rgb_feats[i], all_thm_feats[i]]).to('cpu').detach().numpy()
        all_feats[i] = np.reshape(val, (n_images*2, -1))
    labels = ['RGB']*n_images + ['Thermal']*n_images

    # plt.figure(figsize=(30,5))
    sizes = [(100,100), (60,30), (120,90), (60,30), (120,50)]
    for i in range(5):
        ax= plt.subplot(2,5,start+i)
        plt.axis('off')
        # plt.xlim(-sizes[i][0],sizes[i][0])
        # plt.ylim(-sizes[i][1],sizes[i][1])
        # tsne_vis(features=all_feats[i], labels=labels, title=f'FPN {i+1}: {all_rgb_feats[i].shape[2:]}')
        tsne_vis(features=all_feats[4-i], labels=labels, title='')

    # plt.show()
    model.cuda()
    return ax





# TODO: visualize boxes from RGB vs thermal detectors
def visualize_boxes(model:RGBxThermalDetector, eval_dataloader, gt_dir, thermal_only=False):
    gt_files = os.listdir(gt_dir)
        
    gt_idx = 0
    for batch in tqdm(eval_dataloader):
        # get input on gpu
        x_rgb, x_thermal, _ = batch
        x_rgb, x_thermal = x_rgb.cuda(), x_thermal.cuda()

        # extract features
        rgb_feats_res, rgb_feats_fpn, _ = model.rgb_detector(x_rgb)
        thm_feats_res, thm_feats_fpn, _ = model.thermal_detector(model.thermal_prelayer(x_thermal))

        if thermal_only:
            thm_detections = model.thermal_detector.predict(x_thermal, thm_feats_fpn)
            rgb_detections = model.rgb_detector.predict(x_rgb, rgb_feats_fpn)

        
            


        # TODO: fusing trial - get mask (inverted), multiply with thm features
        masks = []
        bsz = len(x_rgb)
        for img_idx in range(bsz):
            mask = get_mask(skimage.color.rgb2gray(x_rgb[img_idx].cpu().detach().numpy().transpose(1,2,0)))
            mask = np.logical_not(mask)
            masks.append(mask)

        combined_features = []
        for i,sz in enumerate([64, 32, 16, 8, 4]):
            # get mask at current size
            curr_feat = torch.zeros((bsz, 256, sz, sz)).cuda()
            for img_idx in range(len(x_rgb)):
                mask_rsz = resize(masks[img_idx], (sz,sz))

                # weighted average of features at background locations
                new_feat = rgb_feats_fpn[i][img_idx]
                thm_feat = thm_feats_fpn[i][img_idx]
                # new_feat = (new_feat+ thm_feat*2)/3.0
                new_feat[:,mask_rsz] = (new_feat[:,mask_rsz] + thm_feat[:,mask_rsz]*2)/3.0

                curr_feat[img_idx, ...] = new_feat

            combined_features.append(curr_feat)

        print([r.shape for r in combined_features])
        print([r.shape for r in rgb_feats_fpn])

        combined_detections = model.rgb_detector.predict(x_rgb, combined_features)

        for img_index in range(len(thm_detections)):
        # img_index = 7
            thm_detection = thm_detections[img_index]
            rgb_detection = rgb_detections[img_index]
            com_detection = combined_detections[img_index]
            thm_boxes = thm_detection['boxes'].cpu().detach().numpy()
            rgb_boxes = rgb_detection['boxes'].cpu().detach().numpy()
            com_boxes = com_detection['boxes'].cpu().detach().numpy()

            gt_file = gt_files[gt_idx]
            gt_idx += 1
            tree = ET.parse(f'{gt_dir}/{gt_file}')
            root = tree.getroot()
            objs = root[6:]
            gt_boxes = []
            for obj in objs:
                xmin = int(obj[4][0].text)
                ymin = int(obj[4][1].text)
                xmax = int(obj[4][2].text)
                ymax = int(obj[4][3].text)
                gt_boxes.append([xmin, ymin, xmax, ymax])

            fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
            # fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
            # ax = axes.ravel()
            # ax[0] = plt.subplot(1, 1, 1)
            # ax[0] = plt.subplot(1, 3, 1)
            # ax[1] = plt.subplot(1, 3, 2)
            # ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])
            
            # ax[2].imshow(mask, cmap=plt.cm.gray)
            

            ax.imshow(x_thermal[img_index].cpu().detach().numpy().transpose((1,2,0)), cmap='gray')
            for box in thm_boxes:
                xmin, ymin, xmax, ymax = box
                ax.add_patch(patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none'))
            for box in gt_boxes:
                xmin, ymin, xmax, ymax = box
                ax.add_patch(patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='b',facecolor='none'))

            # ax[1].imshow(x_rgb[img_index].cpu().detach().numpy().transpose((1,2,0)))
            # for box in rgb_boxes:
            #     xmin, ymin, xmax, ymax = box
            #     ax[1].add_patch(patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none'))
        
            # ax[2].imshow(x_rgb[img_index].cpu().detach().numpy().transpose((1,2,0)))
            # for box in com_boxes:
            #     xmin, ymin, xmax, ymax = box
            #     ax[2].add_patch(patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none'))
        
            plt.show()


def save_predictions(model:RGBxThermalDetector, eval_dataloader, exp_dir, mode='combined'):
    save_dir = f"{exp_dir}/predictions_{mode}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # for saving
    img_i = 0
    img_names = os.listdir('data/test/rgb')

    for batch in tqdm(eval_dataloader):
        # get input on gpu
        x_rgb, x_thermal, masks = batch
        x_rgb, x_thermal = x_rgb.cuda(), x_thermal.cuda()

        # extract features
        rgb_feats_res, rgb_feats_fpn, _ = model.rgb_detector(x_rgb)
        thm_feats_res, thm_feats_fpn, _ = model.thermal_detector(model.thermal_prelayer(x_thermal))

        # fusing - get mask (inverted), multiply with thm features
        if 'combined' in mode:
            # get loaded masks as numpy -- next block for computing mask from scratch instead of using loaded
            bsz = len(x_rgb)
            masks = masks.cpu().detach().numpy()
            masks = np.logical_not(masks)

            # plt.subplot(121), plt.imshow(masks[0])
            # masks = []
            # for img_idx in range(bsz):
            #     mask = get_mask(skimage.color.rgb2gray(x_rgb[img_idx].cpu().detach().numpy().transpose(1,2,0)))
            #     mask = np.logical_not(mask)
            #     masks.append(mask)
            # plt.subplot(122), plt.imshow(masks[0])
            # plt.show()


            scales = [1,1,0.5,0.2,0.2] # NOTE: best
            # scales = [1,1,1,1,1]
            # scales = [1,1,0.8,0.6,0.4]
            # scales = [1,0.5,0.2,0.05,0.01]
            # scales = [0.4,0.6,0.8,1.0,1.0]
            # scales = [0.2,0.2,0.5,1.0,1.0]
            # scales = [0.01,0.05,0.2,0.5,1.0]
            factor = 5.0

            combined_features = []
            for i,sz in enumerate([64, 32, 16, 8, 4]):
                
                # get mask at current size
                curr_feat = torch.zeros((bsz, 256, sz, sz)).cuda()
                for img_idx in range(len(x_rgb)):
                    mask_rsz = resize(masks[img_idx], (sz,sz))

                    # weighted average of features at background locations
                    new_feat = rgb_feats_fpn[i][img_idx]
                    thm_feat = thm_feats_fpn[i][img_idx]
                    
                    # if sz > 30:
                    new_feat[:,mask_rsz] = (new_feat[:,mask_rsz] + thm_feat[:,mask_rsz]*factor*scales[i])/(1.0+factor*scales[i]) # background boosting
                    # new_feat[:,mask_rsz] = (new_feat[:,mask_rsz] + thm_feat[:,mask_rsz]*scales[i]) # background boosting
                    # new_feat[:,mask_rsz] = thm_feat[:,mask_rsz]

                    curr_feat[img_idx, ...] = new_feat

                combined_features.append(curr_feat)

            detections = model.rgb_detector.predict(x_rgb, combined_features)

        elif mode == 'rgb':
            detections = model.rgb_detector.predict(x_rgb, rgb_feats_fpn)

        elif mode == 'thermal':
            detections = model.thermal_detector.predict(x_thermal, thm_feats_fpn)


        for detection in detections:
            boxes = detection['boxes'].cpu().detach().numpy()
            scores = detection['scores'].cpu().detach().numpy()

            save_name = f'{save_dir}/{img_names[img_i].split(".")[0]}.txt'
            f = open(save_name, "w")

            for score, box in zip(scores, boxes):
                f.write(f"Tree {score} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n")
            
            img_i += 1
            f.close()


def save_difficult_predictions(model:RGBxThermalDetector, eval_dataloader, exp_dir, overlap=50, mode='combined'):
    save_dir = f"{exp_dir}/predictions_{mode}_diff_o={overlap}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # for saving
    img_i = 0
    img_names = os.listdir('data/test/rgb')


    for batch in tqdm(eval_dataloader):
        # get input on gpu
        x_rgb, x_thermal, masks = batch
        x_rgb, x_thermal = x_rgb.cuda(), x_thermal.cuda()

        # extract features
        rgb_feats_res, rgb_feats_fpn, _ = model.rgb_detector(x_rgb)
        thm_feats_res, thm_feats_fpn, _ = model.thermal_detector(model.thermal_prelayer(x_thermal))

        masks = masks.cpu().detach().numpy()
        masks = np.logical_not(masks)

        # fusing - get mask (inverted), multiply with thm features
        if 'combined' in mode:
            # get loaded masks as numpy -- next block for computing mask from scratch instead of using loaded
            bsz = len(x_rgb)
            

            scales = [1,1,0.5,0.2,0.2] # NOTE: best
            # scales = [1,1,1,1,1]
            # scales = [1,1,0.8,0.6,0.4]
            # scales = [1,0.5,0.2,0.05,0.01]
            # scales = [0.4,0.6,0.8,1.0,1.0]
            # scales = [0.2,0.2,0.5,1.0,1.0]
            # scales = [0.01,0.05,0.2,0.5,1.0]
            factor = 5.0
            # factor - 0.5
            # factor = 1.0
            # factor = 2.5
            # factor = 7.5

            combined_features = []
            for i,sz in enumerate([64, 32, 16, 8, 4]):
                
                # get mask at current size
                curr_feat = torch.zeros((bsz, 256, sz, sz)).cuda()
                for img_idx in range(len(x_rgb)):
                    mask_rsz = resize(masks[img_idx], (sz,sz))

                    # weighted average of features at background locations
                    new_feat = rgb_feats_fpn[i][img_idx]
                    thm_feat = thm_feats_fpn[i][img_idx]
                    
                    # if sz > 30:
                    new_feat[:,mask_rsz] = (new_feat[:,mask_rsz] + thm_feat[:,mask_rsz]*factor*scales[i])/(1.0+factor*scales[i]) # background boosting
                    # new_feat[:,mask_rsz] = (new_feat[:,mask_rsz] + thm_feat[:,mask_rsz]*scales[i]) # background boosting
                    # new_feat[:,mask_rsz] = thm_feat[:,mask_rsz]

                    curr_feat[img_idx, ...] = new_feat

                combined_features.append(curr_feat)

            detections = model.rgb_detector.predict(x_rgb, combined_features)

        elif mode == 'rgb':
            detections = model.rgb_detector.predict(x_rgb, rgb_feats_fpn)

        elif mode == 'thermal':
            detections = model.thermal_detector.predict(x_thermal, thm_feats_fpn)


        

        for mask, detection in zip(masks, detections):
            boxes = detection['boxes'].cpu().detach().numpy()
            scores = detection['scores'].cpu().detach().numpy()
            # if 'combined' not in mode:
            #     mask = mask.cpu().detach().numpy()

            save_name = f'{save_dir}/{img_names[img_i].split(".")[0]}.txt'
            f = open(save_name, "w")


            # only keep boxes with >given% overlap with background regions
            count = 0
            for score, box in zip(scores, boxes):
                xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                fg_sum = np.sum(mask[ymin:ymax, xmin:xmax]) # counts 1s within the box basically
                area = (xmax-xmin) * (ymax-ymin)            # total pixels in box
                curr_overlap = fg_sum/area*100              # simple formula
                
                if curr_overlap >= overlap:   
                    count += 1
                    f.write(f"Tree {score} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n")

            img_i += 1
            f.close()

            # remove file if no preds?
            # if count == 0:
            #     os.remove(save_name)


        


def visualize_predictions_side_by_side(img_dir, gt_dir, rgb_pred_dir, combined_pred_dir, mask_dir):
    
    img_names = os.listdir(img_dir)
    mask_files = os.listdir(mask_dir)

    gt_files = os.listdir(gt_dir)
    rgb_pred_files = os.listdir(rgb_pred_dir)
    combined_pred_files = os.listdir(combined_pred_dir)

    for i in range(len(img_names)):
        if i < 7:
            continue
        # elif i > 8:
        #     break

        # image & mask
        img = skimage.io.imread(f'{img_dir}/{img_names[i]}')
        mask = torch.load(f"{mask_dir}/{mask_files[i]}")

        # get groundtruth boxes
        gt_file = gt_files[i]
        tree = ET.parse(f'{gt_dir}/{gt_file}')
        root = tree.getroot()
        objs = root[6:]
        gt_boxes = []
        for obj in objs:
            xmin = int(obj[4][0].text)
            ymin = int(obj[4][1].text)
            xmax = int(obj[4][2].text)
            ymax = int(obj[4][3].text)
            gt_boxes.append([xmin, ymin, xmax, ymax])

        # get RGB boxes
        rgb_boxes = []
        with open(f'{rgb_pred_dir}/{rgb_pred_files[i]}') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            if float(line[1]) < 0.1:
                break
            rgb_boxes.append([int(line[2]), int(line[3]), int(line[4]), int(line[5])])

        # get combined boxes
        combined_boxes = []
        with open(f'{combined_pred_dir}/{combined_pred_files[i]}') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            if float(line[1]) < 0.1:
                break
            combined_boxes.append([int(line[2]), int(line[3]), int(line[4]), int(line[5])])


        def plot_boxes(ax, boxes, color):
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                ax.add_patch(patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=2,edgecolor=color,facecolor='none'))
       

        fig, axes = plt.subplots(ncols=3, figsize=(15, 8))
        ax = axes.ravel()
        ax[0] = plt.subplot(1, 3, 1)
        ax[1] = plt.subplot(1, 3, 2)
        ax[2] = plt.subplot(1, 3, 3)

        ax[2].set_title('FG/BG Mask')
        ax[2].imshow(mask)

        ax[0].set_title('RGB Only')
        ax[0].imshow(img)
        plot_boxes(ax[0], gt_boxes, color='b')
        plot_boxes(ax[0], rgb_boxes, color='r')

        ax[1].set_title('Combined')
        ax[1].imshow(img)
        plot_boxes(ax[1], gt_boxes, color='b')
        plot_boxes(ax[1], boxes=combined_boxes, color='r')

        plt.show()
        




    

def compare_model_weights(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


def compare_model_outputs(our:UTDARetinanet, loader):

    img_dir = 'others/brighten_predictions_imgs'
    img_names = os.listdir(img_dir)

    df_rgb_model = df_main.deepforest()
    df_rgb_model.use_release()
    df_rgb_model.cuda()
    df_rgb_model.eval()

    save_dir = 'others/shadow_removal/brighten'
    # os.makedirs(save_dir)

    from PIL import Image
    for name in tqdm(img_names):
        # img = skimage.io.imread(f'{img_dir}/{name[:-4]}.png')

        # x_rgb, x_thm, mask = next(iter(loader))
        # x_rgb = x_rgb[0].cuda()
        # print(x_rgb[0].unsqueeze(0))

        
        # img_tensor = torch.Tensor(img.transpose((2,0,1))).unsqueeze(0).cuda()

        # rgb_feats_res, rgb_feats_fpn, _ = our(img_tensor)
        # our_out = our.predict(img_tensor, rgb_feats_fpn)
        # boxes = our_out[0]['boxes'].cpu().detach().numpy()
        # scores = our_out[0]['scores'].cpu().detach().numpy()
        # # print(len(our_out[0]['boxes']))
        # # print(our_out[0]['boxes'].cpu().detach().numpy().astype(np.uint8))

        # save_name = f'others/baseline/our_rgb/{name.split(".")[0]}.txt'
        # f = open(save_name, "w")
        # for score, box in zip(scores, boxes):
        #     f.write(f"Tree {score} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n")
        # f.close()

        # im = Image.open(f'{img_dir}/{name[:-4]}.png')
        df_out = df_rgb_model.predict_image(path=f'{img_dir}/{name[:-4]}.png', return_plot=False)
        # df_out = df_rgb_model.predict_image(np.array(im).astype('float32'), return_plot=False)

        save_name = f'{save_dir}/{name.split(".")[0]}.txt'
        f = open(save_name, "w")
        for row in df_out.iterrows():
            box = [row[1]['xmin'], row[1]['ymin'], row[1]['xmax'], row[1]['ymax']]
            f.write(f"Tree {row[1]['score']} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n")
        f.close()

        # our_out_2 = our.predict(x_rgb)
        # print(our_out_2)
        # exit()





def eval_encDec(model:UNet, eval_dataloader, exp_dir, mode='combined', rgb_model=None, df_model=None, skip_vis=False):
    save_dir = f"{exp_dir}/predictions_{mode}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # for saving
    img_i = 0
    img_names = os.listdir('data/test/rgb')

    for batch in tqdm(eval_dataloader):
        # get input on gpu
        x_rgb, x_thermal, masks = batch
        x_rgb, x_thermal, masks = x_rgb.cuda(), x_thermal.cuda(), masks.cuda()

        out = model.convert(x_thermal)

        out = out.cpu().detach().numpy().transpose((0,2,3,1))
        x_rgb = x_rgb.cpu().detach().numpy().transpose((0,2,3,1))
        x_thermal = x_thermal.cpu().detach().numpy().transpose((0,2,3,1))
        masks = torch.logical_not(masks).unsqueeze(1).repeat(1,3,1,1)
        masks = masks.cpu().detach().numpy().transpose((0,2,3,1))

        max_, min_= np.amax(out), np.amin(out)
        out = (out-min_)/(max_-min_)


        x_combined = out.copy()
        # x_combined[masks] = (x_rgb[masks] * out[masks]) 

        x_combined = x_combined / np.amax(x_combined) 

        out = (out*255.0).astype(np.uint8) 
        x_rgb = (x_rgb*255.0).astype(np.uint8) 
        x_combined = (x_combined*255.0).astype(np.uint8) 
        

        for i in range(len(x_rgb)):
            pred_combined = df_model.predict_image(x_combined[i], return_plot=False)
            save_name = f'{save_dir}/{img_names[img_i].split(".")[0]}.txt'
            f = open(save_name, "w")

            for row in pred_combined.iterrows():
                box = [row[1]['xmin'], row[1]['ymin'], row[1]['xmax'], row[1]['ymax']]
                f.write(f"Tree {row[1]['score']} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n")

            img_i += 1
            f.close()
            

        if skip_vis:
            continue

         

        for i in range(len(x_rgb)):
            pred_rgb = df_model.predict_image(x_rgb[i], return_plot=True, color=(0,0,255))
            pred_combined = df_model.predict_image(x_combined[i], return_plot=True, color=(0,0,255))
            
            plt.figure(figsize=(11,7))
            plt.subplot(231), plt.imshow(pred_rgb[:,:,::-1]),                 plt.title('Original RGB (w/ detections)')
            plt.subplot(232), plt.imshow(x_thermal[i], cmap='gray'),plt.title('Original Thermal')
            plt.subplot(233), plt.imshow(out[i]),                   plt.title('Converted Thermal')
            plt.subplot(234), plt.imshow(masks[i,...,0], cmap='gray'), plt.title('Mask (Bg: White)')
            plt.subplot(235), plt.imshow(x_combined[i]),            plt.title('Fused')
            if pred_combined is not None:
                plt.subplot(236), plt.imshow(pred_combined[:,:,::-1]), plt.title('Fused Detections')
            else:
                plt.subplot(236), plt.title('No Detections')
            plt.show()



def save_figs_for_diagrams(rgb_dir, thm_dir, mask_dir, mask_other_dir, gt_dir, rgb_pred_dir, combined_pred_dir):
    ''' Saves plt.figs for (rgb image, thermal image, mask (box), mask (morph) predictions on rgb, predictions on fused, ground truth)'''

    img_names = os.listdir(rgb_dir)
    mask_files = os.listdir(mask_dir)
    other_mask_files = os.listdir(mask_other_dir)

    gt_files = os.listdir(gt_dir)
    rgb_pred_files = os.listdir(rgb_pred_dir)
    combined_pred_files = os.listdir(combined_pred_dir)

    # outdir = './plots/combined_boxes'
    outdir = './plots/diagram_figures'
    # try: os.makedirs(outdir)
    # except: pass


    for i in tqdm(range(len(img_names))):
        curr_outdir = f"{outdir}/{img_names[i].split('.')[0]}"
        # curr_outdir = outdir
        try: os.makedirs(curr_outdir)
        except: pass


        # image & mask
        img = skimage.io.imread(f'{rgb_dir}/{img_names[i]}')
        thm_img = skimage.io.imread(f'{thm_dir}/{img_names[i]}')
        mask = torch.load(f"{mask_dir}/{mask_files[i]}")
        mask_other = torch.load(f'{mask_other_dir}/{other_mask_files[i]}')

        # get groundtruth boxes
        gt_file = gt_files[i]
        tree = ET.parse(f'{gt_dir}/{gt_file}')
        root = tree.getroot()
        objs = root[6:]
        gt_boxes = []
        for obj in objs:
            xmin = int(obj[4][0].text)
            ymin = int(obj[4][1].text)
            xmax = int(obj[4][2].text)
            ymax = int(obj[4][3].text)
            gt_boxes.append([xmin, ymin, xmax, ymax])

        # get RGB boxes
        rgb_boxes = []
        with open(f'{rgb_pred_dir}/{rgb_pred_files[i]}') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            if float(line[1]) < 0.1:
                break
            rgb_boxes.append([int(line[2]), int(line[3]), int(line[4]), int(line[5])])

        # get therm boxes: TODO

        # get combined boxes
        combined_boxes = []
        with open(f'{combined_pred_dir}/{combined_pred_files[i]}') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            if float(line[1]) < 0.1:
                break
            combined_boxes.append([int(line[2]), int(line[3]), int(line[4]), int(line[5])])

            


        def plot_boxes(ax, boxes, color):
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                ax.add_patch(patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=0.5,edgecolor=color,facecolor='none'))
    

        def get_fig_ax():
            sizes = np.shape(img)
            height = float(sizes[0])
            width = float(sizes[1])
            
            fig = plt.figure(dpi=600)
            fig.set_size_inches(width/height, 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            return fig, ax

        
        fig, ax = get_fig_ax()
        ax.imshow(mask, cmap='gray')
        plt.savefig(f'{curr_outdir}/mask.png', bbox_inches=0)
        plt.close()

        fig, ax = get_fig_ax()
        ax.imshow(np.logical_not(mask), cmap='gray')
        plt.savefig(f'{curr_outdir}/bg mask.png', bbox_inches=0)
        plt.close()

        fig, ax = get_fig_ax()
        ax.imshow(mask_other, cmap='gray')
        plt.savefig(f'{curr_outdir}/mask other.png', bbox_inches=0)
        plt.close()

        fig, ax = get_fig_ax()
        ax.imshow(np.logical_not(mask_other), cmap='gray')
        plt.savefig(f'{curr_outdir}/bg mask other.png', bbox_inches=0)
        plt.close()

        fig, ax = get_fig_ax()
        ax.imshow(img)
        plt.savefig(f'{curr_outdir}/rgb.png', bbox_inches=0)
        plt.close()

        # ax[6].set_title('Thermal')
        fig, ax = get_fig_ax()
        ax.imshow(thm_img, cmap='gray')
        plt.savefig(f'{curr_outdir}/thermal.png', bbox_inches=0)
        plt.close()

        # ax[0].set_title('RGB Prediction')
        fig, ax = get_fig_ax()
        ax.imshow(img)
        # plot_boxes(ax[0], gt_boxes, color='b')
        plot_boxes(ax, rgb_boxes, color='b')
        plt.savefig(f'{curr_outdir}/rgb_pred_new.png', bbox_inches=0)
        plt.close()
#
        # ax[1].set_title('Combined')
        fig, ax = get_fig_ax()
        ax.axis('off')
        ax.imshow(img)
        # plot_boxes(ax[1], gt_boxes, color='b')
        plot_boxes(ax, boxes=combined_boxes, color='purple')
        plt.savefig(f'{curr_outdir}/combined_pred.png', bbox_inches=0)
        # plt.savefig(f'{curr_outdir}/{img_names[i]}.png', bbox_inches=0)
        plt.close()

        # ax[2].set_title('Ground')
        fig, ax = get_fig_ax()
        ax.axis('off')
        ax.imshow(img)
        plot_boxes(ax, gt_boxes, color='orange')
        plt.savefig(f'{curr_outdir}/ground truth.png', bbox_inches=0)
        plt.close()
        

        # ax = axes.ravel()
        # ax[0] = plt.subplot(1, 9, 1)
        # ax[1] = plt.subplot(1, 9, 2)
        # ax[2] = plt.subplot(1, 9, 3)
        # ax[3] = plt.subplot(1, 9, 4)
        # ax[4] = plt.subplot(1, 9, 5)
        # ax[5] = plt.subplot(1, 9, 6)
        # ax[6] = plt.subplot(1, 9, 7)
        # ax[7] = plt.subplot(1, 9, 8)
        # ax[8] = plt.subplot(1, 9, 9)

        # for ax_ in ax:
        #     ax_.axis('off')


        

        # ax[3].set_title('RGB')
        

        # if i == 0:
        #     plt.close()



def qual_results(rgb_dir, thm_dir, mask_dir, gt_dir, rgb_pred_dir, rgb_pred_dir_2, therm_pred_dir, combined_pred_dir):#, metafuse_dir, metafuse_preds, bright_dir, bright_preds, dat_preds):
    ''' Saves plt.figs for (rgb image, thermal image, mask (box), mask (morph) predictions on rgb, predictions on fused, ground truth)'''

    img_names = os.listdir(rgb_dir)
    mask_files = os.listdir(mask_dir)

    gt_files = os.listdir(gt_dir)
    rgb_pred_files = os.listdir(rgb_pred_dir)
    combined_pred_files = os.listdir(combined_pred_dir)

    # outdir = './plots/combined_boxes'
    outdir = './plots/qual_results_supp'
    try: os.makedirs(outdir)
    except: pass

    # i_s = [1,2,4,8,18,24]
    # xlims = ([0,220],[0,220],[220,420],[0,200],[280,500],[150,350])
    # ylims = ([0,220],[140,360],[100,300],[300,500],[150,370],[0,200])

    i_s = [9, 11, 20, 30, 52, 4] 
    xlims = ([0,250],[70,290],[0,200],[0,250], [265,500], [220,420])
    ylims = ([250,0],[390,170],[460,260],[300,50], [365,130], [85,285])


    # for i in tqdm(range(len(img_names))):
    for iidx,i in enumerate(tqdm(i_s)):

        # image & mask
        img = skimage.io.imread(f'{rgb_dir}/{img_names[i]}')
        thm_img = skimage.io.imread(f'{thm_dir}/{img_names[i]}')
        mask = torch.load(f"{mask_dir}/{mask_files[i]}")

        # get groundtruth boxes
        gt_file = gt_files[i]
        tree = ET.parse(f'{gt_dir}/{gt_file}')
        root = tree.getroot()
        objs = root[6:]
        gt_boxes = []
        for obj in objs:
            xmin = int(obj[4][0].text)
            ymin = int(obj[4][1].text)
            xmax = int(obj[4][2].text)
            ymax = int(obj[4][3].text)
            gt_boxes.append([xmin, ymin, xmax, ymax])

        # get RGB boxes
        rgb_file = f'{rgb_pred_dir_2}/{rgb_pred_files[i]}'
        if not os.path.exists(rgb_file):
            rgb_file = f'{rgb_pred_dir}/{rgb_pred_files[i]}'
        rgb_boxes = []
        with open(f'{rgb_pred_dir}/{rgb_pred_files[i]}') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            if float(line[1]) < 0.1:
                break
            rgb_boxes.append([int(line[2]), int(line[3]), int(line[4]), int(line[5])])

        # get therm boxes: TODO
        therm_boxes = []
        with open(f'{therm_pred_dir}/{rgb_pred_files[i]}') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            if float(line[1]) < 0.1:
                break
            therm_boxes.append([int(line[2]), int(line[3]), int(line[4]), int(line[5])])

        # get combined boxes
        combined_boxes = []
        with open(f'{combined_pred_dir}/{combined_pred_files[i]}') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            if float(line[1]) < 0.1:
                break
            combined_boxes.append([int(line[2]), int(line[3]), int(line[4]), int(line[5])])

            


        def plot_boxes(ax, boxes, color):
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                ax.add_patch(patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor=color,facecolor='none'))
    
        # manual trial
        if False:
            fig, axes = plt.subplots(3,3, figsize=(15, 15))
            ax = axes.ravel()
            for idx, ax_ in enumerate(ax):
                if idx != 2:
                    ax_.axis('off')
                else:
                # Hide X and Y axes label marks
                    ax_.xaxis.set_tick_params(labelbottom=False)
                    ax_.yaxis.set_tick_params(labelleft=False)
                    ax_.set_xticks([])
                    ax_.set_yticks([])
            ax[0] = plt.subplot(3, 3, 1)
            ax[1] = plt.subplot(3, 3, 2)
            ax[2] = plt.subplot(3, 3, 3)
            ax[3] = plt.subplot(3, 3, 4)
            ax[4] = plt.subplot(3, 3, 5)
            ax[5] = plt.subplot(3, 3, 6)
            ax[6] = plt.subplot(3, 3, 7)


            # RGB image, thermal image, mask, GT boxes (orange), RGB pred (blue), Thermal pred (red), Fused pred (purple)
            plt.suptitle(f'Image {i}: {img_names[i]}')
            ax[5].imshow(img), ax[5].set_title('rgb')
            ax[1].imshow(thm_img, cmap='gray'), ax[1].set_title('thm')
            ax[2].imshow(mask, cmap='gray'), ax[2].set_title('mask')
            ax[3].imshow(img), plot_boxes(ax[3], rgb_boxes, color='blue'), ax[3].set_title('rgb_pred')
            ax[4].imshow(thm_img, cmap='gray'), plot_boxes(ax[4], therm_boxes, color='red'), ax[4].set_title('thm_pred')
            ax[0].imshow(img), plot_boxes(ax[0], combined_boxes, color='purple'), ax[0].set_title('fused_pred')
            ax[6].imshow(img), plot_boxes(ax[6], gt_boxes, color='orange'), ax[6].set_title('gt')
            plt.show()

        # actual saving
        else:
            fig, axes = plt.subplots(7,1, figsize=(15, 15)) 
            fig.subplots_adjust(hspace=0.05)
            ax = axes.ravel()
            for idx, ax_ in enumerate(ax):
                if idx != 2:
                    ax_.axis('off')
                else:
                # Hide X and Y axes label marks
                    ax_.xaxis.set_tick_params(labelbottom=False)
                    ax_.yaxis.set_tick_params(labelleft=False)
                    ax_.set_xticks([])
                    ax_.set_yticks([])
                ax_.set_xlim(xlims[iidx])
                ax_.set_ylim(ylims[iidx])
            ax[0] = plt.subplot(7, 1, 1)
            ax[1] = plt.subplot(7, 1, 2)
            ax[2] = plt.subplot(7, 1, 3)
            ax[3] = plt.subplot(7, 1, 4)
            ax[4] = plt.subplot(7, 1, 5)
            ax[5] = plt.subplot(7, 1, 6)
            ax[6] = plt.subplot(7, 1, 7)


            # RGB image, thermal image, mask, GT boxes (orange), RGB pred (blue), Thermal pred (red), Fused pred (purple)
            print(img.shape)
            print(img_names[i])
            ax[0].imshow(img)
            ax[1].imshow(thm_img, cmap='gray')
            ax[2].imshow(mask, cmap='gray')
            ax[3].imshow(img), plot_boxes(ax[3], rgb_boxes, color='blue')
            ax[4].imshow(thm_img, cmap='gray'), plot_boxes(ax[4], therm_boxes, color='red')
            ax[5].imshow(img), plot_boxes(ax[5], combined_boxes, color='violet')
            ax[6].imshow(img), plot_boxes(ax[6], gt_boxes, color='orange')
            plt.savefig(f'{outdir}/{i}.pdf', bbox_inches='tight', pad_inches=0.5 if iidx == 0 else 0.1)
            # plt.show()


        

        # ax[3].set_title('RGB')
        

        # if i == 0:
        #     plt.close()


def disp_thermal_prelayer_output(model:RGBxThermalDetector, dataloader):

    for batch in tqdm(eval_dataloader):
        # get input on gpu
        x_rgb, x_thermal, masks = batch
        x_rgb, x_thermal = x_rgb.cuda(), x_thermal.cuda()

        pre_output = model.return_prelayer_output(x_thermal)
        print(pre_output.shape)

        for i in range(len(batch)):
            img = x_rgb[i].detach().cpu().numpy().transpose((1,2,0))
            thm = x_thermal[i].detach().cpu().numpy().transpose((1,2,0))
            out = pre_output[i].detach().cpu().numpy().transpose((1,2,0))

            max_, min_ = np.amax(out), np.amin(out)
            out = (out - min_) / (max_ - min_)
        
            
            

            plt.subplot(131), plt.imshow(img)
            plt.subplot(132), plt.imshow(out)
            plt.subplot(133), plt.imshow(thm, cmap='gray')
            plt.show()


def display_masks(n=5):
    plt.figure()


if __name__ == "__main__":
    pass

    # from data_utils import save_binary_masks
    # save_binary_masks('data/test/rgb', 'data/test/masks')

    # # data loaders
    # boxes = False
    # boxes_txt = '_boxes' if boxes else ''
    # eval_dataset = RGBxThermalTreeCrownDataset(root='data', split='val', masks_dir='masks'+boxes_txt) # TODO: if needed change masks dir to masks_boxes
    # eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    # # # get trained model
    # # exp_dir = 'output/cynthia_fieldwork'
    # # exp_dir = 'output/others/df_thermal'
    # # exp_dir = 'output/ours - res+disc together(grl), then fpn'
    # exp_dir = f'output/ablation/alignment_scales/[1,1,1,1,1]'
    exp_dir = f'output/ours - single stage (scaled)'
    # if not os.path.exists(exp_dir):
    #     os.makedirs(exp_dir)
    # # trained_model_path = f'output/ours - single stage (scaled)/checkpoints/chkpt_10000.pt'
    # trained_model_path = f'{exp_dir}/checkpoints/chkpt_10000.pt'
    
    # trained_model_path = f'{exp_dir}/checkpoints/chkpt_10000.pt'
    # trained_model_path = 'output/ours - three stages/checkpoints/chkpt_25.pt'

    # # NOTE: remove path
    # chkpt_df = torch.load('others/fine_tuned/thermal_10.pt')
    # state_dict = chkpt_df['state_dict']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k,v in state_dict.items():
    #     new_state_dict[k[6:]] = v.clone()
    #     # del state_dict[k]


    # model = RGBxThermalDetector(state_dict=new_state_dict)
    # model = RGBxThermalDetector()
    # ckpt = torch.load(trained_model_path)
    # model.load_state_dict(ckpt['model_state_dict'], strict=False)
    # # model.load_state_dict(new_state_dict, strict=False)
    # model.cuda()
    # model.eval()

    # olap = 85
    # save_difficult_predictions(model, eval_dataloader, exp_dir, overlap=olap, mode='combined'+boxes_txt)
    # exit()


    # disp_thermal_prelayer_output(model, eval_dataloader)

    

    # other_trained_model_path = f'{exp_dir}/checkpoints/chkpt_9000.pt'
    # model_2 = RGBxThermalDetector()
    # ckpt = torch.load(other_trained_model_path)
    # model_2.load_state_dict(ckpt['model_state_dict'], strict=False)
    # model_2.cuda()
    # model_2.eval()



    
    # df_rgb_model = df_main.deepforest()
    # df_rgb_model.use_release()
    # df_rgb_model.cuda()



    # compare_model_weights(model, model_2)
    # exit()

    # print(str(model.thermal_detector.state_dict()) == str(model_2.thermal_detector.state_dict()) )

    # exit()

    # visualize_res_features(model, eval_dataloader)
    # fig = plt.figure(figsize=(18,5))
    # visualize_fpn_features(model, eval_dataloader, start=6)
    # # model.load_state_dict(ckpt['model_state_dict'], strict=False)
    # model = RGBxThermalDetector()
    # model.cuda()
    # model.eval()
    # ax = visualize_fpn_features(model, eval_dataloader, start=1)
    # print(ax.get_legend_handles_labels())
    # fig.subplots_adjust(hspace=0.4, wspace=0.6)
    # red_patch = patches.Patch(edgecolor='red',  color='red', facecolor='red', label='Thermal')
    # blue_patch = patches.Patch(edgecolor='blue', color='blue', facecolor='blue', label='RGB')

    # fig.legend([blue_patch, red_patch], labels = ['RGB', 'Thermal'], loc='upper center', ncol=2, bbox_to_anchor=(0.5,0.075), fontsize="24")
    # plt.savefig('plots/feature_vis.pdf', bbox_inches='tight', pad_inches=1)
    # plt.show()
    
    # exit()
    # visualize_boxes(model, eval_dataloader, gt_dir='data/test/gt_annotations_difficult', thermal_only=True)


    # save_predictions(model, eval_dataloader, exp_dir, mode='rgb'+boxes_txt)
    # save_predictions(model, eval_dataloader, exp_dir, mode='thermal'+boxes_txt)
    # save_predictions(model, eval_dataloader, exp_dir, mode='combined'+boxes_txt)
    # save_predictions(model, eval_dataloader, exp_dir, mode='thermal'+boxes_txt)
    # visualize_predictions_side_by_side(f'{eval_dataset.rgb_dir}', 'data/test/gt_annotations_new',  # TODO: remove the _2
    #                                    f'{exp_dir}/predictions_rgb',
    #                                    f'{exp_dir}/predictions_combined'+boxes_txt, 
    #                                    f'data/test/masks'+boxes_txt)
    # exit()

    # # NOTE: qual results
    qual_results('data/test/rgb', 'data/test/thermal', 'data/test/masks', 'data/test/gt_annotations_new', 
                f'{exp_dir}/predictions_rgb',
                'output/only updating prelayer and disc/different optims, 1x (best!)/predictions_rgb',
                f'{exp_dir}/predictions_thermal',
                f'{exp_dir}/predictions_combined')
    
    olap = 85
    # save_difficult_predictions(model, eval_dataloader, exp_dir, overlap=olap, mode='rgb')
    # # save_difficult_predictions(model, eval_dataloader, exp_dir, overlap=85, mode='thermal')
    # save_difficult_predictions(model, eval_dataloader, exp_dir, overlap=olap, mode='combined'+boxes_txt)
    # visualize_predictions_side_by_side('data/test/rgb', 'data/test/gt_annotations_difficult',  # TODO: remove the _2
    #                                    f'{exp_dir}/predictions_rgb_diff_o={olap}',
    #                                    f'{exp_dir}/predictions_combined'+boxes_txt+f'_diff_o={olap}', 
    #                                    f'data/test/masks'+boxes_txt)
    # save_difficult_predictions(model, eval_dataloader, exp_dir, overlap=85, mode='combined'+boxes_txt)
    # save_difficult_predictions(model, eval_dataloader, exp_dir, overlap=90, mode='combined'+boxes_txt)
    
    

    ## Vanilla Deepforest:
    # df_rgb_model = df_main.deepforest()
    # df_rgb_model.use_release()
    # df_rgb_model.cuda()
    # path='D:\ShadowFormer-main\shadowformer_shadowless_image.pt'    
    # img = torch.load(path)
    # img = skimage.img_as_ubyte(img)
    # plot = df_rgb_model.predict_image(img, return_plot=True)
    # # plot = df_rgb_model.predict_image(path=path, return_plot=True)
    # plt.imshow(plot[:,:,::-1])
    # plt.show()

    # compare_model_weights(rgb_model, model.rgb_detector)


    ## encoder decoder structure
    # df_rgb_model = df_main.deepforest()
    # df_rgb_model.use_release()
    # # df_rgb_model.cuda()
    # model = RGBxThermalDetector()
    # model.cuda()
    # model.eval()
    # rgb_model = model.rgb_detector

    # exp_dir = 'output/enc_dec/only content loss'
    # trained_model_path = f'{exp_dir}/checkpoints/final_10000.pt'
    # model = UNet(1, 3)
    # ckpt = torch.load(trained_model_path)
    # model.load_state_dict(ckpt['model_state_dict'], strict=False)
    # # model.load_state_dict(ckpt['G_model_state_dict'])
    # model.cuda()
    # model.eval()
    # eval_encDec(model, eval_dataloader, exp_dir, mode='combined', 
    #             rgb_model=rgb_model, df_model=df_rgb_model, skip_vis=True)
    # visualize_predictions_side_by_side('data/test/rgb', 'data/test/gt_annotations',  # TODO: remove the _2
    #                                    f'output/tm=whole/da=single_theirs/fm=bg_avg - burnin+training(best) - alternate freezing/predictions_rgb',
    #                                    f'{exp_dir}/predictions_combined', 
    #                                    f'data/test/masks')

    # save_figs_for_diagrams('data/test/rgb', 'data/test/thermal', 'data/test/masks', 'data/test/masks_boxes', 'data/test/gt_annotations_new', f'{exp_dir}/predictions_rgb',
    #                                     f'{exp_dir}/predictions_combined')

    
    # df_rgb_model.cuda()
    
    # model = RGBxThermalDetector('df_retinanet.pt')
    # model.eval()
    # # ckpt = torch.load('output/tm=whole/da=single_theirs/fm=bg_avg - burnin+training(best) - alternate freezing/predictions_rgb/')
    # # model.load_state_dict(ckpt['model_state_dict'], strict=False)
    # compare_model_outputs(model.rgb_detector, eval_dataloader)
