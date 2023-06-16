import gc
import os

import numpy as np
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.utils.generic_utils import Progbar

from methods.wrappers import calculate_relevancemap
from utils.metrics import calculate_morf_curve, input_relevance_correlation
from utils.mnist import load_and_prepare_data
from utils.model import get_models


def run_bulk_mnist(methods, net, variant, inverted=False, morfnum=250, batch_size=None, calc_morf=True, calc_pcc=True, calc_scc=True, **kwargs):
    # Prepare labels and paths
    invlbl = {True: 'INV', False: ''}
    model_id = '{}MNIST{}{}'.format(net, variant, invlbl[inverted])

    # Load models
    model_w_softmax, model_w_o_softmax = get_models(model_id)

    # Derive paths
    path_results = 'results/MNIST_{}'.format(model_id)

    # Create directory for results if absent
    os.makedirs(path_results, exist_ok=True)

    # Load and prepare data
    (_, _), (x_test, _) = load_and_prepare_data(variant, inverted)
    nfiles = len(x_test)

    # Iterate over methods
    for m in methods:
        # Create progress bar
        pb = Progbar(nfiles, width=50, verbose=1, interval=0.05)

        # Create placeholder array for MoRF curves
        if calc_morf:
            morf_curves = np.zeros((nfiles, morfnum+1))

        # Create placeholder array for pearson correlation values
        if calc_pcc:
            pcc_values = np.zeros(nfiles)

        # Create placeholder array for spearman correlation values
        if calc_scc:
            scc_values = np.zeros(nfiles)

        for i, x in enumerate(x_test):
            # Re-load models in case of guided grad cam to keep graph small (as new models are created in every iteration)
            if m.startswith('guided_grad_cam'):
                clear_session()
                model_w_softmax, model_w_o_softmax = get_models(model_id)

            # Update progress bar
            pb.update(i + 1)

            # Calculate relevancemap
            R = calculate_relevancemap(m, x, model_w_o_softmax, **kwargs)

            # Calculate MoRF curve
            if calc_morf:
                morf_curve = calculate_morf_curve(x, R, model_w_softmax, morfnum, batch_size=batch_size)
                morf_curves[i] = morf_curve

            # Calculate correlation
            if calc_pcc:
                pcc_values[i] = input_relevance_correlation(x, R, variant='pearson')

            if calc_scc:
                scc_values[i] = input_relevance_correlation(x, R, variant='spearman')

            # Perform garbage collection
            gc.collect()

        # Save results for methods iteration
        if calc_morf:
            np.save('{}/morfcurves_{}.npy'.format(path_results, m), morf_curves)
        if calc_pcc:
            np.save('{}/pccvalues_{}.npy'.format(path_results, m), pcc_values)
        if calc_scc:
            np.save('{}/sccvalues_{}.npy'.format(path_results, m), scc_values)
