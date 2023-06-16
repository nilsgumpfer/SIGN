import gc
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.utils.generic_utils import Progbar

from methods.wrappers import calculate_relevancemap
from utils.file import remove_filetype
from utils.metrics import input_relevance_correlation, pearsonr, spearmanr
from utils.model import get_models
from utils.preprocessing import get_image, preprocess_image


def run_single_PCC(method, dataset_id, model_id, filename, figdim=5, **kwargs):
    # Load models
    model_w_softmax, model_w_o_softmax = get_models(model_id)

    # Derive path
    path_plots = 'plots/{}_{}'.format(dataset_id, model_id)
    path_saveto = '{}/{}_PCC_({}).png'.format(path_plots, remove_filetype(filename), method)

    # Create directory for plots if absent
    os.makedirs(path_plots, exist_ok=True)

    # Load and preprocess image
    img, x = get_image(filename, dataset_id, expand_dims=False)

    # Calculate relevancemap
    R = calculate_relevancemap(method, x, model_w_o_softmax, **kwargs)

    # Calculate correlation
    xvalues, hvalues, pcc_value = input_relevance_correlation(x, R, return_values=True)
    print('PCC: {:.2f}'.format(pcc_value))

    # Plot values
    plt.figure(figsize=(figdim, figdim))
    plt.scatter(xvalues, hvalues, cmap='Blues_r', s=1, norm=plt.Normalize(vmax=80))
    plt.xlabel('|x|')
    plt.ylabel('|R|')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    xmax = np.max(np.abs(np.ravel(preprocess_image(np.array([[[0., 0., 0.], [128., 128., 128.], [255., 255., 255.]]])))))
    plt.xlim((0, xmax * 1.01))
    plt.xticks([0, np.max(xvalues) / 2, xmax * 1.01])
    plt.ylim((0, np.max(hvalues) * 1.01))
    plt.yticks([0, np.max(hvalues) / 2, np.max(hvalues) * 1.01])
    plt.title('{} (PCC = {:.2f})'.format(method, pcc_value))

    plt.tight_layout()
    plt.savefig(path_saveto)
    plt.close()

    # Perform garbage collection and session clearance
    gc.collect()
    clear_session()

    return path_saveto


def run_dataset_n_CC_map(method, dataset_id, model_id, n, figdim=5, res=150, **kwargs):
    # Load models
    model_w_softmax, model_w_o_softmax = get_models(model_id)

    # Derive paths
    path_dataset = 'data/datasets/{}'.format(dataset_id)
    path_plots = 'plots/{}_{}'.format(dataset_id, model_id)
    path_saveto = '{}/{}_n{}_PCC_({})'.format(path_plots, dataset_id, n, method)

    # Create directory for plots if absent
    os.makedirs(path_plots, exist_ok=True)

    # Select files to be used
    filenames = sorted(os.listdir(path_dataset))[:n]

    # Define placeholders
    _, xtmp = get_image(filenames[0], dataset_id, expand_dims=False)
    shape = np.shape(xtmp)
    X, Rs, PCC, SCC = np.zeros((n, shape[0], shape[1], shape[2])), np.zeros((n, shape[0], shape[1], shape[2])), np.zeros(n), np.zeros(n)

    # Create progress bar
    pb = Progbar(n, width=50, verbose=1, interval=0.05)

    # Collect relevancemaps and inputs
    for i, f in enumerate(filenames):
        # Update pb
        pb.update(i+1)

        # Load and preprocess image
        img, x = get_image(f, dataset_id, expand_dims=False)

        # Calculate relevancemap
        R = calculate_relevancemap(method, x, model_w_o_softmax, **kwargs)

        # Store x and h
        X[i] = x
        Rs[i] = R

        # Store CCs
        PCC[i] = pearsonr(np.ravel(np.abs(x)), np.ravel(np.abs(R)))
        SCC[i] = spearmanr(np.ravel(np.abs(x)), np.ravel(np.abs(R)))

    # Calculate input-relevance correlation
    xvalues = np.ravel(np.abs(X))
    Rvalues = np.ravel(np.abs(Rs))
    del X
    del Rs
    gc.collect()

    # Derive map
    themap = calcmap(xvalues, Rvalues, res=res)
    print('Calculated map')

    xvalues_max = np.max(xvalues)
    hvalues_max = np.max(Rvalues)
    del xvalues
    del Rvalues
    gc.collect()

    # Plot values
    plt.figure(figsize=(figdim*1.1, figdim))
    plt.imshow(themap, interpolation=None, cmap=LinearSegmentedColormap.from_list(name='WB', colors=[(1, 1, 1), (0, 102/255, 204/255)]), clim=(0, 1))
    plt.xlabel(r'$|x_i|$')
    plt.ylabel(r'$|R_i|$')
    xmax = np.max(np.abs(np.ravel(preprocess_image(np.array([[[0., 0., 0.], [128., 128., 128.], [255., 255., 255.]]])))))
    if hvalues_max < 1.0:
        yticks = ['0.0'] + [np.format_float_scientific(n, precision=1, exp_digits=1) for n in [hvalues_max / 2, hvalues_max * 1.01]]
    else:
        yticks = ['{:.1f}'.format(n) for n in [0.0, hvalues_max / 2, hvalues_max * 1.01]]
    xticks = np.round([0, xvalues_max / 2, xmax * 1.01], 1)
    plt.xticks([0, int(res/2), res], labels=xticks)
    plt.yticks([res, int(res/2), 0], labels=yticks)
    plt.gca().add_artist(AnchoredText(r'\noindent $\overline{\text{PCC}}$' + ' = {:.2f}'.format(np.mean(PCC)) + r'\\ $\overline{\text{SCC}}$' + ' = {:.2f}'.format(np.mean(SCC)), frameon=False, loc='upper right'))

    plt.tight_layout()
    plt.savefig(path_saveto)
    plt.close()

    # Perform garbage collection and session clearance
    gc.collect()
    clear_session()

    return path_saveto


def calcmap(xvalues, hvalues, res=200):
    xvalues = np.array(np.round((xvalues / np.max(xvalues)) * res), dtype=int)
    hvalues = np.array(np.round(((hvalues / np.max(hvalues)) * -res) + res), dtype=int)
    themap = np.zeros((res+1, res+1))
    themap[hvalues, xvalues] = 1

    return themap
