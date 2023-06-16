import gc
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from tensorflow.python.keras.backend import clear_session

from methods.wrappers import calculate_relevancemap
from utils.file import remove_filetype
from utils.innv import aggregate_and_normalize_relevancemap_rgb
from utils.metrics import generate_position_tuples, generate_perturbed_inputs, \
    calculate_predictions, generate_perturbed_heatmaps, calculate_aoc
from utils.model import get_models
from utils.preprocessing import get_image, reverse_preprocess_image


def run_single_MoRF(method, dataset_id, model_id, filename, brightness=1.0, contrast=1.0, morfnum=250, batch_size=10, figdim=4, **kwargs):
    positions = [0, 0.05, 0.2, 1]
    snapshot_idxs = [int(morfnum * p) for p in positions]
    ticklabels = ['{}%'.format(int(idx*100)) for idx in positions]

    # Load models
    model_w_softmax, model_w_o_softmax = get_models(model_id)

    # Derive path
    path_plots = 'plots/{}_{}'.format(dataset_id, model_id)

    # Create directory for plots if absent
    os.makedirs(path_plots, exist_ok=True)

    # Load and preprocess image
    img, x = get_image(filename, dataset_id, brightness=brightness, contrast=contrast, expand_dims=False)

    # Calculate relevancemap
    R = calculate_relevancemap(method, x, model_w_o_softmax, **kwargs)

    # Aggregate relevancemap on pixel-level
    H = np.array(aggregate_and_normalize_relevancemap_rgb(R))

    # Generate sorted list of pixel positions based on their relevance (highest first)
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
        ax[1].imshow(reverse_preprocess_image(perturbed_inputs[idx]))
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
            ax[i].set_xlabel('224px')
            ax[i].set_ylabel('224px')
            ax[i].set_xticks([])
            ax[i].set_yticks([])

    # Add AOC to last plot
    axs[-1][2].add_artist(AnchoredText('MoRF AOC = {:.4f}'.format(morf_aoc), frameon=False, loc='upper right'))

    plt.tight_layout()
    plt.savefig('{}/{}_MoRF_({})'.format(path_plots, remove_filetype(filename), method))
    plt.close()

    # Perform garbage collection
    clear_session()
    gc.collect()

    return morf_aoc
