import gc
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.backend import clear_session

from methods.wrappers import calculate_relevancemap
from utils.file import remove_filetype
from utils.innv import aggregate_and_normalize_relevancemap_rgb
from utils.metrics import calculate_morf_curve, input_relevance_correlation, calculate_aoc
from utils.misc import pretty_method_name
from utils.model import get_models, decode_prediction, decode_prediction_top_n
from utils.preprocessing import get_image


def run_single(method, dataset_id, model_id, filename, morfnum=250, batch_size=10, figdim=5, **kwargs):
    # Load models
    model_w_softmax, model_w_o_softmax = get_models(model_id)

    # Derive path
    path_plots = 'plots/{}_{}'.format(dataset_id, model_id)

    # Create directory for plots if absent
    os.makedirs(path_plots, exist_ok=True)

    # Load and preprocess image
    img, x = get_image(filename, dataset_id, expand_dims=False)

    # Calculate relevancemap
    R = calculate_relevancemap(method, x, model_w_o_softmax, **kwargs)

    # Calculate MoRF curve and AOC
    if morfnum is not None:
        morf_curve = calculate_morf_curve(x, R, model_w_softmax, morfnum, batch_size=batch_size)
        morf_aoc = calculate_aoc(morf_curve)
    else:
        morf_aoc = 0

    # Calculate correlation
    pcc_value = input_relevance_correlation(x, R, variant='pearson')
    scc_value = input_relevance_correlation(x, R, variant='spearman')

    # Predict
    pred_idx, pred_class, pred_prob = decode_prediction(model_w_softmax.predict(np.array([x])), model_id)

    # Generate, print and save stats
    stats = {'morf_aoc': round(morf_aoc, 2), 'pcc_value': round(float(pcc_value), 2), 'scc_value': round(float(scc_value), 2), 'pred_class': pred_class, 'pred_prob': round(float(pred_prob), 2)}
    print(stats)
    with open('{}/{}_({}).json'.format(path_plots, remove_filetype(filename), method), mode='w') as fp:
        json.dump(stats, fp)

    # Aggregate and normalize relevancemap for visualization
    H = aggregate_and_normalize_relevancemap_rgb(R)

    # Plot relevancemap
    plt.figure(figsize=(figdim*1.23, figdim))
    plt.imshow(H, cmap='seismic', clim=(-1, 1))

    cb = plt.colorbar(ticks=[-1, 0, 1])
    cb.set_label(r'$\mathcal{H}_{H,W}$', rotation=0, horizontalalignment="left", verticalalignment="center")
    plt.xlabel('224px')
    plt.ylabel('224px')
    plt.xticks([])
    plt.yticks([])
    plt.title(pretty_method_name(method))

    plt.tight_layout()
    plt.savefig('{}/{}_({}).png'.format(path_plots, remove_filetype(filename), method))
    plt.close()

    # Perform garbage collection
    clear_session()
    gc.collect()


def run_single_explain_top_n(methods, dataset_id, model_id, filename, top_n=3, morfnum=250, batch_size=10, brightness=1.0, contrast=1.0, figdim=5, **kwargs):
    # Load models
    model_w_softmax, model_w_o_softmax = get_models(model_id)

    # Derive path
    path_plots = 'plots/{}_{}'.format(dataset_id, model_id)

    # Create directory for plots if absent
    os.makedirs(path_plots, exist_ok=True)

    # Load and preprocess image
    img, x = get_image(filename, dataset_id, brightness=brightness, contrast=contrast, expand_dims=False)

    # Predict on model
    IDX, PC, PB = decode_prediction_top_n(model_w_softmax.predict(np.array([x])), model_id, top_n=top_n)

    # Prepare plot
    fig, axs = plt.subplots(nrows=top_n, ncols=len(methods) + 1, figsize=((len(methods) + 1) * figdim, top_n * figdim))

    # Plot inputs
    for r in range(top_n):
        axs[r][0].matshow(img)

    # Input title
    axs[0][0].set_title('Input image', fontsize=36, rotation=-22.5, horizontalalignment="right", verticalalignment="bottom")

    # Plot relevancemaps method-wise
    for c, m in enumerate(methods):
        print(m)

        # Calculate relevancemaps
        Rs = [calculate_relevancemap(m, np.array(x), model_w_o_softmax, neuron_selection=int(idx)) for idx in IDX]

        # Method title
        axs[0][c + 1].set_title(pretty_method_name(m, wo_params=True), fontsize=36, rotation=-22.5, horizontalalignment="right", verticalalignment="bottom")

        for r in range(top_n):
            # Calculate MoRF curve and AOC
            morf_curve = calculate_morf_curve(x, Rs[r], model_w_softmax, morfnum, batch_size=batch_size, idx=IDX[r])
            morf_aoc = calculate_aoc(morf_curve)

            # Calculate correlation
            pcc_value = input_relevance_correlation(x, Rs[r], variant='pearson')
            scc_value = input_relevance_correlation(x, Rs[r], variant='spearman')

            # Derive stats string
            stats = 'AOC: {:.4f}, PCC: {:.2f}, SCC: {:.2f}'.format(morf_aoc, pcc_value, scc_value)
            print(stats)

            # Aggregate and normalize relevancemap for visualization
            H = aggregate_and_normalize_relevancemap_rgb(Rs[r])
            axs[r][c + 1].matshow(H, cmap='seismic', clim=(-1, 1))
            axs[r][c + 1].set_xlabel(stats, fontsize=18)

    for r in range(top_n):
        axs[r][0].set_xlabel('pred: {}\n prob: {:.2f}'.format(PC[r], PB[r]), fontsize=18)

    for ax in np.ravel(axs):
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig('{}/{}_{}_{}_top_{}'.format(path_plots, remove_filetype(filename), dataset_id, model_id, top_n), orientation="landscape", bbox_inches="tight")
    plt.close()

    clear_session()
    gc.collect()


def run_single_explain_classes(methods, dataset_id, model_id, filename, class_idxs, morfnum=250, batch_size=10, brightness=1.0, contrast=1.0, figdim=5, **kwargs):
    # Load models
    model_w_softmax, model_w_o_softmax = get_models(model_id)

    # Derive path
    path_plots = 'plots/{}_{}'.format(dataset_id, model_id)

    # Create directory for plots if absent
    os.makedirs(path_plots, exist_ok=True)

    # Load and preprocess image
    img, x = get_image(filename, dataset_id, brightness=brightness, contrast=contrast, expand_dims=False)

    # Load labels
    with open('data/models/{}labels.json'.format(model_id)) as fp:
        labels_idxs = json.load(fp)

    # Predict on model
    y_pred = model_w_softmax.predict(np.array([x]))[0]

    # Prepare plot
    fig, axs = plt.subplots(nrows=len(class_idxs), ncols=len(methods) + 1, figsize=((len(methods) + 1) * figdim, len(class_idxs) * figdim))

    # Plot inputs
    for r in range(len(class_idxs)):
        axs[r][0].matshow(img)

    # Input title
    axs[0][0].set_title('Input image', fontsize=36, rotation=-22.5, horizontalalignment="right", verticalalignment="bottom")

    # Plot relevancemaps method-wise
    for c, m in enumerate(methods):
        print(m)

        # Calculate relevancemaps
        Rs = [calculate_relevancemap(m, np.array(x), model_w_o_softmax, neuron_selection=int(idx)) for idx in class_idxs]

        # Method title
        axs[0][c + 1].set_title(pretty_method_name(m, wo_params=True), fontsize=36, rotation=-22.5, horizontalalignment="right", verticalalignment="bottom")

        for r, idx in enumerate(class_idxs):
            # Calculate MoRF curve and AOC
            morf_curve = calculate_morf_curve(x, Rs[r], model_w_softmax, morfnum, batch_size=batch_size, idx=idx)
            morf_aoc = calculate_aoc(morf_curve)

            # Calculate correlation
            pcc_value = input_relevance_correlation(x, Rs[r], variant='pearson')
            scc_value = input_relevance_correlation(x, Rs[r], variant='spearman')

            # Derive stats string
            # stats = 'AOC: {:.4f}, PCC: {:.2f}, SCC: {:.2f}'.format(morf_aoc, pcc_value, scc_value)
            stats = 'AOC: {:.4f}'.format(morf_aoc)
            print(stats)

            # Aggregate and normalize relevancemap for visualization
            H = aggregate_and_normalize_relevancemap_rgb(Rs[r])
            axs[r][c + 1].matshow(H, cmap='seismic', clim=(-1, 1))
            axs[r][c + 1].set_xlabel(stats, fontsize=24)

    for r, c in enumerate(class_idxs):
        axs[r][0].set_xlabel('pred: {}\n prob: {:.2f}'.format(labels_idxs[str(c)], y_pred[c]), fontsize=18)

    for ax in np.ravel(axs):
        ax.set_xticks([])
        ax.set_yticks([])

    suffix = ''
    for idx in class_idxs:
        suffix += '{}_'.format(idx)
    suffix = suffix[:-1]

    plt.savefig('{}/{}_{}_{}_{}'.format(path_plots, remove_filetype(filename), dataset_id, model_id, suffix), orientation="landscape", bbox_inches="tight")
    plt.close()

    clear_session()
    gc.collect()
