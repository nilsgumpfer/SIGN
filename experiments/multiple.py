import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.python.keras.backend import clear_session

from methods.wrappers import calculate_relevancemap
from utils.innv import aggregate_and_normalize_relevancemap_rgb
from utils.metrics import calculate_morf_curve, input_relevance_correlation, calculate_aoc
from utils.misc import pretty_method_name
from utils.model import get_models, decode_prediction
from utils.preprocessing import get_image


def run_multiple(methods, dataset_id, model_id, filenames, group_name, morfnum=250, batch_size=10, brightness=1.0, contrast=1.0, figdim=5, **kwargs):
    # Load models
    model_w_softmax, model_w_o_softmax = get_models(model_id)

    # Derive path
    path_plots = 'plots/{}_{}'.format(dataset_id, model_id)

    # Create directory for plots if absent
    os.makedirs(path_plots, exist_ok=True)

    IMG, X, PC, PB = [], [], [], []
    for f in filenames:
        # Load and preprocess image
        img, x = get_image(f, dataset_id, brightness=brightness, contrast=contrast, expand_dims=False)
        IMG.append(img)
        X.append(x)

        # Predict on model
        pred_idx, pred_class, pred_prob = decode_prediction(model_w_softmax.predict(np.array([x])), model_id)
        PC.append(pred_class)
        PB.append(pred_prob)

    # Prepare plot
    fig, axs = plt.subplots(nrows=len(filenames), ncols=len(methods) + 1, figsize=((len(methods) + 1) * figdim, len(filenames) * figdim))

    # Plot inputs
    for r, img in enumerate(IMG):
        axs[r][0].matshow(img)

    # Input title
    axs[0][0].set_title('Input image', fontsize=36, rotation=-22.5, horizontalalignment="right", verticalalignment="bottom")

    # collected = []

    # Plot relevancemaps method-wise
    for c, m in enumerate(methods):
        print(m)

        # Calculate relevancemaps
        Rs = [calculate_relevancemap(m, np.array(x), model_w_o_softmax) for x in X]

        # Method title
        axs[0][c + 1].set_title(pretty_method_name(m, wo_params=True), fontsize=36, rotation=-22.5, horizontalalignment="right", verticalalignment="bottom")

        for r in range(len(filenames)):
            # Calculate MoRF curve and AOC
            morf_curve = calculate_morf_curve(X[r], Rs[r], model_w_softmax, morfnum, batch_size=batch_size)
            morf_aoc = calculate_aoc(morf_curve)

            # Calculate correlation
            pcc_value = input_relevance_correlation(X[r], Rs[r], variant='pearson')
            scc_value = input_relevance_correlation(X[r], Rs[r], variant='spearman')

            # Derive stats string
            # collected.append({'filename': filenames[r], 'method': m, 'AOC': morf_aoc, 'PCC': pcc_value, 'SCC': scc_value})
            # stats = 'AOC: {:.4f}, PCC: {:.2f}, SCC: {:.2f}'.format(morf_aoc, pcc_value, scc_value)
            stats = 'AOC: {:.4f}'.format(morf_aoc)
            print(stats)

            # Aggregate and normalize relevancemap for visualization
            H = aggregate_and_normalize_relevancemap_rgb(Rs[r])
            axs[r][c + 1].matshow(H, cmap='seismic', clim=(-1, 1))
            axs[r][c + 1].set_xlabel(stats, fontsize=24)

    for r in range(len(filenames)):
        axs[r][0].set_xlabel('pred: {}\n prob: {:.2f}'.format(PC[r], PB[r]), fontsize=18)

    for ax in np.ravel(axs):
        ax.set_xticks([])
        ax.set_yticks([])

    # pd.DataFrame(collected).to_excel('{}/{}_metrics.xlsx'.format(path_plots, group_name), index=False)

    plt.savefig('{}/{}_{}_g{}'.format(path_plots, dataset_id, model_id, group_name), orientation="landscape", bbox_inches="tight")
    plt.close()

    clear_session()
    gc.collect()



