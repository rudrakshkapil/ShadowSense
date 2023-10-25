
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import scipy.stats as stats
import numpy as np
import colorcet as cc

from matplotlib import cm
import pandas as pd
import joypy
import seaborn as sns

from mpl_toolkits.axes_grid1 import axes_grid

from data_utils import *


def box_counts_per_image():
    count, total, all_counts, all_totals = count_difficult('data/test/gt_annotations_new')
    # plt.hist(all_totals, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    # plt.xlabel()
    # plt.show()
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

    # fig.add_subplot(111, frameon=False)
    fig.tight_layout()

    # hide tick and tick label of the big axis
    # plt.title('')
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.ylabel('Frequency', fontsize=10)
    plt.savefig('BBox Histograms.pdf', bbox_inches='tight')



def box_areas_per_image(gt_dir):
    # count, total, all_counts, all_totals = count_difficult('data/test/gt_annotations_new')
    # fig, axes = plt.subplots(2,1,sharex=True, sharey=True)

    all_areas, other_areas, difficult_areas = get_areas(gt_dir)
    # plt.subplot(131)

    # plt.plot(np.arange(0,100), all_areas)
    # plt.show()
    # exit()
    
    # plt.violinplot([all_areas, difficult_areas, other_areas],showmedians=True)
    
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

    plt.figure(figsize=(6,4))
    all_dims, other_dims, difficult_dims = get_HW(gt_dir)
    all_dims = np.array(all_dims)
    all_dims = all_dims[np.sum(all_dims,axis=1)>10]

    difficult_dims = np.array(difficult_dims)
    difficult_dims = difficult_dims[np.sum(difficult_dims,axis=1)>10]



    df = pd.DataFrame({'h':all_dims[:,0],'w':all_dims[:,1],'Box Type':['Non-Difficult']*len(all_dims)})
    df = pd.DataFrame({'h':all_dims[:,0],'w':all_dims[:,1]})
    # df2 = pd.DataFrame({'h':difficult_dims[:,0],'w':difficult_dims[:,1],'Box Type':['Difficult']*len(difficult_dims)})
    # df = pd.concat([df,df2])

    # for dim in all_dims:
    #     print(dim)
    #     df.append(dim, ignore_index=True)
    print(df.head())


    PALETTE = sns.color_palette('pastel', 100)

    g = sns.histplot(
        df,
        x="w",
        y="h",
        # hue='Box Type',
        cbar=True, cbar_kws=dict(shrink=.75),
        # y_label_key="relative_height",
        # y_label_name="Height (in % of image)",
        # title=self.title,
        # x_lim=(0, 100),
        # y_lim=(0, 100),
        # x_ticks_rotation=None,
        # labels_key="split",
        # individual_plots_key="split",
        # tight_layout=False,
        # sharey=True,
        color='Blue'
    )
    # sns.move_legend(g, "lower right")

    
    plt.xlabel("Bounding Box Width (pixels)")
    plt.ylabel("Bounding Box Height (pixels)")
    plt.xlim(-10,170)
    plt.ylim(-10,170)
    # plt.legend(loc='lower right')
    plt.title("Bounding Box Dimensions Across All 500x500px Images")
    plt.savefig("plots/BBox Dimensions.pdf", bbox_inches='tight')


def brightness_plot():
    # get_brightness_by_date()
    df = pd.read_csv('plots/brightness.csv')
    palette = sns.color_palette(cc.glasbey, n_colors=14)
    sns.displot(df, x="Brightness (units)", hue="Flight Date", kind="kde", palette=palette)
    # plt.title('KDE Plot for Average Pixel Brightness (in LAB space) Across All Flight Dates', loc='center')
    plt.savefig('plots/brightness_by_date.pdf', bbox_inches='tight')

def plot_box_brightnesses():
    all_areas, other_areas, difficult_areas, full_image = \
        get_box_brightness('data/test/gt_annotations_new', 'data/test/rgb', 'data/test/thermal')
    plt.savefig('plots/brightness_by_date.pdf', bbox_inches='tight')

    all_areas = np.array(all_areas)
    other_areas = np.array(other_areas)
    difficult_areas = np.array(difficult_areas)
    full_image = np.array(full_image)

    # df = pd.Series(all, name="NA").to_frame().join(pd.Series(HG, name="HG"))

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
        # 'Whole Images RGB':full_image[:,0], 
    #     # 'Whole Images Thermal':full_image[:,1],
    #     # 'All Boxes RGB':all_areas[:,0], 
    #     # 'All Boxes Thermal':full_image[:,1],
    #     'Other Boxes RGB':other_areas[:,0], 
    #     'Other Boxes Thermal':other_areas[:,1],
    #     'Difficult Boxes RGB':difficult_areas[:,0], 
    #     'Difficult Boxes Thermal':difficult_areas[:,1],
    # })
    
    sns.boxplot(df, x = 'Brightness', y='Category', hue="Domain", palette=["b", "r"])
    plt.show()


def example_images():

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
        
        
           

            # ag[j].axes('off')
            # ag.

    plt.subplots_adjust(wspace=0.1, hspace=0.01)
    plt.savefig('plots/example_images.pdf', bbox_inches='tight')

    


if __name__ == "__main__":
    # box_dimensions('data/test/gt_annotations_new')
    # box_areas_per_image('data/test/gt_annotations_new')

    # plot_box_brightnesses()
    # example_images()
    brightness_plot()