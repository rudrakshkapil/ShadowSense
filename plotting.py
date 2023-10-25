"""
Utility script for plotting diagrams used in manuscript
"""

import numpy as np
import colorcet as cc
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import axes_grid

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage.transform import resize

from data_utils import *
from df_repo.deepforest import main as df_main


def tsne_vis(features, labels, palette=['blue','red'], title="T-SNE plot", perplexity=30, pca=False):
    """ t-SNE visualization of features """
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
    """ Visualize features of ResNet in feature space """
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
    """ Visualize features of FPN in feature space """

    all_rgb_feats = None
    all_thm_feats = None
    model.cpu()

    for batch in tqdm(eval_dataloader):
        # get input on gpu
        x_rgb, x_thermal, _ = batch

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
        tsne_vis(features=all_feats[4-i], labels=labels, title='')

    # plt.show()
    model.cuda()
    return ax


def visualize_boxes(model, eval_dataloader, gt_dir, thermal_only=False):
    """ visualize boxes from RGB vs thermal detectors """
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
            ax.imshow(x_thermal[img_index].cpu().detach().numpy().transpose((1,2,0)), cmap='gray')
            for box in thm_boxes:
                xmin, ymin, xmax, ymax = box
                ax.add_patch(patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none'))
            for box in gt_boxes:
                xmin, ymin, xmax, ymax = box
                ax.add_patch(patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='b',facecolor='none'))

            plt.show()


def box_counts_per_image():
    """ Plot number of boxes per image as 2D histogram """
    count, total, all_counts, all_totals = count_difficult('data/test/gt_annotations_new')
    fig, axes = plt.subplots(2,1,sharex=True, sharey=True)
    
    def plot_hist(x, ax, title):
        n, bins, patches = ax.hist(x, bins=np.arange(min(x),max(x)+2)-0.5,facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)

        n = n.astype('int') # it MUST be integer
        # Good old loop. Choose colormap of your taste
        for i in range(len(patches)):
            patches[i].set_facecolor(plt.cm.magma(n[i]/(max(n)+2)))

        # Add annotation
        x_label = 'Difficult ' if 'Difficult' in title else ""
            
        ax.set_title(title, fontsize=15, color='w')
        ax.xaxis.set_tick_params(labelbottom=True)
        loc = plticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel(f'Number of {x_label}Bounding Boxes per Image', fontsize=12)

        
    plot_hist(all_totals, axes[0], 'Distribution for All Bounding Boxes')
    plot_hist(all_counts, axes[1], 'Distribution for Difficult Bounding Boxes')

    fig.tight_layout()
    plt.savefig('BBox Histograms.pdf', bbox_inches='tight')



def box_areas_per_image(gt_dir):
    """ Plot area of boxes in each image (separated by difficult/non0difficult) as whisker plots """
    
    all_areas, other_areas, difficult_areas = get_areas(gt_dir)
    
    # Set up the figure and axis
    fig, ax = plt.subplots(1, 1)

    # Create a list of the data to be plotted
    data = [all_areas, difficult_areas, other_areas]

    # Set the colors for the violins based on the category
    colors = ['Blue', 'Orange', 'Teal']

    # Create the violin plot
    plots = ax.violinplot(data, vert=True, showmedians=True, showextrema=True, widths=0.75)

    # Set the color of the violin patches
    for pc, color in zip(plots['bodies'], colors):
        pc.set_facecolor(color)

    # Set the color of the median lines
    plots['cmedians'].set_colors(colors)

    # Set the labels
    # ax.set_yticks([1, 2, 3], labels=['All Boxes', 'Difficult Boxes', 'Non-difficult Boxes'])
    ax.set_xticks([1, 2, 3], labels=['All Boxes', 'Difficult Boxes', 'Non-difficult Boxes'], fontsize=10)
    plt.title("Distribution of Bounding Box Areas", fontsize=12)
    plt.ylabel("Bounding Box Area (% of Total Image Area)", fontsize=10)
    plt.savefig('plots/BBox Areas.pdf', bbox_inches='tight')


def box_dimensions(gt_dir):
    """ Plot dimensons of boxes """

    plt.figure(figsize=(6,4))
    all_dims, other_dims, difficult_dims = get_HW(gt_dir)
    all_dims = np.array(all_dims)
    all_dims = all_dims[np.sum(all_dims,axis=1)>10]

    difficult_dims = np.array(difficult_dims)
    difficult_dims = difficult_dims[np.sum(difficult_dims,axis=1)>10]

    df = pd.DataFrame({'h':all_dims[:,0],'w':all_dims[:,1],'Box Type':['Non-Difficult']*len(all_dims)})
    df = pd.DataFrame({'h':all_dims[:,0],'w':all_dims[:,1]})

    PALETTE = sns.color_palette('pastel', 100)

    g = sns.histplot(
        df,
        x="w",
        y="h",
        cbar=True, cbar_kws=dict(shrink=.75),
        color='Blue'
    )
    
    plt.xlabel("Bounding Box Width (pixels)")
    plt.ylabel("Bounding Box Height (pixels)")
    plt.xlim(-10,170)
    plt.ylim(-10,170)
    plt.title("Bounding Box Dimensions Across All 500x500px Images")
    plt.savefig("plots/BBox Dimensions.pdf", bbox_inches='tight')


def brightness_plot():
    """ Plot KD plots of brightness in each image by flight date """
    # get_brightness_by_date()
    df = pd.read_csv('plots/brightness.csv')
    palette = sns.color_palette(cc.glasbey, n_colors=14)
    sns.displot(df, x="Brightness (units)", hue="Flight Date", kind="kde", palette=palette)
    plt.savefig('plots/brightness_by_date.pdf', bbox_inches='tight')


def plot_box_brightnesses():
    """ Plot brightness of boxes by difficulty """
    
    all_areas, other_areas, difficult_areas, full_image = \
        get_box_brightness('data/test/gt_annotations_new', 'data/test/rgb', 'data/test/thermal')
    plt.savefig('plots/brightness_by_date.pdf', bbox_inches='tight')

    all_areas = np.array(all_areas)
    other_areas = np.array(other_areas)
    difficult_areas = np.array(difficult_areas)
    full_image = np.array(full_image)

    df = pandas.DataFrame({
        'Brightness': full_image[:,0],
        'Category': ['Whole Image'] * len(full_image),
        'Domain': ['RGB'] * len(full_image)
    })
    df = pandas.concat([df, pandas.DataFrame({
        'Brightness': all_areas[:,0],
        'Category': ['All Boxes'] * len(all_areas),
        'Domain': ['RGB'] * len(all_areas)
    })])
    df = pandas.concat([df, pandas.DataFrame({
        'Brightness': other_areas[:,0],
        'Category': ['Other Boxes'] * len(other_areas),
        'Domain': ['RGB'] * len(other_areas)
    })])
    df = pandas.concat([df, pandas.DataFrame({
        'Brightness': difficult_areas[:,0],
        'Category': ['Difficult Boxes'] * len(difficult_areas),
        'Domain': ['RGB'] * len(difficult_areas)
    })])
    df = pandas.concat([df, pandas.DataFrame({
        'Brightness': full_image[:,1],
        'Category': ['Whole Image'] * len(full_image),
        'Domain': ['Thermal'] * len(full_image)
    })])
    df = pandas.concat([df, pandas.DataFrame({
        'Brightness': all_areas[:,1],
        'Category': ['All Boxes'] * len(all_areas),
        'Domain': ['Thermal'] * len(all_areas)
    })])
    df = pandas.concat([df, pandas.DataFrame({
        'Brightness': other_areas[:,1],
        'Category': ['Other Boxes'] * len(other_areas),
        'Domain': ['Thermal'] * len(other_areas)
    })])
    df = pandas.concat([df, pandas.DataFrame({
        'Brightness': difficult_areas[:,1],
        'Category': ['Difficult Boxes'] * len(difficult_areas),
        'Domain': ['Thermal'] * len(difficult_areas)
    })])

    df = pandas.concat([df, pandas.DataFrame({
        'Brightness': full_image[:,1],
        'Label': ['Whole Image Thermal'] * len(full_image)
    })])
    
    sns.boxplot(df, x = 'Brightness', y='Category', hue="Domain", palette=["b", "r"])
    plt.show()


def example_images():
    """ Plot an example of an image from each flight date """
    all_image_pairs = get_image_pairs()
    date_names = ['July 20', 'July 26', 'August 9', 'August 17', 'August 30', 'September 9', 'September 15', 'September 23', 'October 4', 'October 6','October 7', 'October 12', 'October 19', 'November 24']
    labels = 'abcdefghijklmn'

    csfont = {'fontname':'Times New Roman'}
    nrows = 7
    ncols = 2
    naxes = 2
    f = plt.figure(figsize=(10,14))
    # plt.tight_layout(w_pad=3)
    for i, pair in enumerate(all_image_pairs):
        ag = axes_grid.Grid(f, (nrows, ncols, i+1), (1, naxes), axes_pad=0.075)
        ag[0].imshow(pair[0])
        ag[0].axis('off')
        ag[1].imshow(pair[1], cmap='gray')
        ag[1].axis('off')
        
        ag[0].text(1500, 1115, f"({labels[i]}) {date_names[i]}", horizontalalignment='center',
     verticalalignment='center', fontsize=9, **csfont)

    plt.subplots_adjust(wspace=0.1, hspace=0.01)
    plt.savefig('plots/example_images.pdf', bbox_inches='tight')


def visualize_predictions_side_by_side(img_dir, gt_dir, rgb_pred_dir, combined_pred_dir, mask_dir):
    """ Visualize predictions of images side by side (RGB, Thermal) """
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
    """ Compare weights of two models """
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


def compare_model_outputs(our, loader):
    """ Compare outputs of model with Deepforest """
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
        df_out = df_rgb_model.predict_image(path=f'{img_dir}/{name[:-4]}.png', return_plot=False)
        save_name = f'{save_dir}/{name.split(".")[0]}.txt'
        f = open(save_name, "w")
        for row in df_out.iterrows():
            box = [row[1]['xmin'], row[1]['ymin'], row[1]['xmax'], row[1]['ymax']]
            f.write(f"Tree {row[1]['score']} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n")
        f.close()



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
        plot_boxes(ax[0], gt_boxes, color='b')
        plot_boxes(ax, rgb_boxes, color='b')
        plt.savefig(f'{curr_outdir}/rgb_pred_new.png', bbox_inches=0)
        plt.close()

        # ax[1].set_title('Combined')
        fig, ax = get_fig_ax()
        ax.axis('off')
        ax.imshow(img)
        plot_boxes(ax[1], gt_boxes, color='b')
        plot_boxes(ax, boxes=combined_boxes, color='purple')
        plt.savefig(f'{curr_outdir}/combined_pred.png', bbox_inches=0)
        plt.savefig(f'{curr_outdir}/{img_names[i]}.png', bbox_inches=0)
        plt.close()

        # ax[2].set_title('Ground')
        fig, ax = get_fig_ax()
        ax.axis('off')
        ax.imshow(img)
        plot_boxes(ax, gt_boxes, color='orange')
        plt.savefig(f'{curr_outdir}/ground truth.png', bbox_inches=0)
        plt.close()



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


def disp_thermal_prelayer_output(model, eval_dataloader):
    """ Display output of thermal prelayer """

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
