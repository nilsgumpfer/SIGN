import gc
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from tensorflow.python.keras.backend import clear_session

from methods.wrappers import calculate_relevancemap
from utils.innv import aggregate_and_normalize_relevancemap_rgb
from utils.metrics import generate_position_tuples, generate_perturbed_inputs, \
    calculate_predictions, generate_perturbed_heatmaps, calculate_aoc
from utils.mnist import load_and_prepare_data
from utils.model import get_models


def run_single_MoRF_mnist(method, net, variant, ind, inverted=False, morfnum=250, batch_size=10, figdim=4, **kwargs):
    # Prepare labels and paths
    invlbl = {True: 'INV', False: ''}
    model_id = '{}MNIST{}{}'.format(net, variant, invlbl[inverted])

    positions = [0, 0.05, 0.2, 1]
    snapshot_idxs = [int(morfnum * p) for p in positions]
    ticklabels = [int(idx*100) for idx in positions]

    # Load models
    model_w_softmax, model_w_o_softmax = get_models(model_id)

    # Derive path
    path_plots = 'plots/MNIST_{}'.format(model_id)

    # Create directory for plots if absent
    os.makedirs(path_plots, exist_ok=True)

    # Load and preprocess image
    (_, _), (x_test, _) = load_and_prepare_data(variant, inverted)
    x = x_test[ind]

    # Derive levels for heatmap contours
    v = np.ravel(x)
    mn, mx = np.min(v), np.max(v)
    levels = [(mn + mx) / 2.0]

    # Calculate relevancemap
    R = calculate_relevancemap(method, x, model_w_o_softmax, **kwargs)

    # Aggregate relevancemap on pixel-level
    H = np.array(aggregate_and_normalize_relevancemap_rgb(R))

    # Generate sorted list of pixel positions based on their mean relevance (highest first)
    position_tuples_sorted = generate_position_tuples(H)

    # Generate perturbed inputs and heatmaps
    perturbed_inputs = generate_perturbed_inputs(x, position_tuples_sorted, num=morfnum)
    perturbed_heatmaps = generate_perturbed_heatmaps(H, position_tuples_sorted, num=morfnum)

    # Calculate predictions / MoRF curve
    prediction_values = calculate_predictions(perturbed_inputs, model_w_softmax, batch_size)

    # Calculate AOC
    morf_aoc = calculate_aoc(prediction_values)

    nrows = len(snapshot_idxs)
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figdim * ncols, figdim * nrows))

    axs[0][0].set_title('perturbed heatmap')
    axs[0][1].set_title('perturbed input')
    axs[0][2].set_title('MoRF curve')

    for ax, idx in zip(axs, snapshot_idxs):
        ax[0].imshow(perturbed_heatmaps[idx], cmap='seismic', clim=(-1, 1))
        ax[1].imshow(perturbed_inputs[idx], cmap='seismic', clim=(-1, 1))
        ax[0].contour(np.squeeze(x, axis=2), levels=levels, linewidths=[3.0], colors='k', linestyles='-')
        ax[1].contour(np.squeeze(x, axis=2), levels=levels, linewidths=[3.0], colors='k', linestyles='-')
        ax[2].plot(prediction_values[:idx+1], c='red', linewidth=3)
        ax[2].set_xlim((0, morfnum))
        ax[2].set_ylim((-0.01, np.max(np.ravel(prediction_values))+0.01))
        ax[2].set_xticks(snapshot_idxs)
        ax[2].set_xticklabels(ticklabels)
        ax[2].set_xlabel(r'perturbed area (\%)')
        ax[2].set_ylabel(r'$f_c(\mathbf{x})$')
        ax[2].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax[2].set_yticklabels([str(t) for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])

    for ax in axs:
        for i in range(2):
            ax[i].set_xlabel('28px')
            ax[i].set_ylabel('28px')
            ax[i].set_xticks([])
            ax[i].set_yticks([])

    # Add AOC to last plot
    axs[-1][2].add_artist(AnchoredText('MoRF AOC = {:.4f}'.format(morf_aoc), frameon=False, loc='upper right'))

    plt.tight_layout()
    plt.savefig('{}/{}_{}_MoRF_({})'.format(path_plots, model_id, ind, method))
    plt.close()

    # Perform garbage collection
    clear_session()
    gc.collect()
