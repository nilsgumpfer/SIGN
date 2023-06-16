import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.activations import linear
from tensorflow.python.keras.backend import clear_session

from utils.mnist import load_or_train_model, load_and_prepare_data
from utils.mnist_xai import compare_classes
from utils.training import enable_reproducibility


def run_simple_MNIST(methods, variant, net, epochs, random_state=0, inverted=False, indices=None, cls1=3, cls2=8, misclassified=False):
    # Reproducibility setup
    enable_reproducibility(random_state)

    # Prepare labels and paths
    invlbl = {True: 'INV', False: ''}
    model_id = '{}MNIST{}{}'.format(net, variant, invlbl[inverted])
    modelpath = 'data/models/{}.h5'.format(model_id)
    path_base = 'plots/MNIST_{}'.format(model_id)
    path = '{}/{}'.format(path_base, model_id)
    os.makedirs(path_base, exist_ok=True)

    # Load and prepare data
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data(variant, inverted)

    # Load or train model
    print(model_id)
    model = load_or_train_model(modelpath, net, x_train, y_train, x_test, y_test, epochs)

    # Predict on model
    y_pred = model.predict(x_test)

    # Remove softmax for explanations
    model.layers[-1].activation = linear

    # Derive misclassified examples
    diff = np.argmax(y_pred, axis=1) - np.argmax(y_test, axis=1)
    mc_idxs = np.arange(start=0, stop=len(y_pred), step=1)[diff != 0]

    # Plot
    if indices is None:
        print('Choose indices from the examples!')
        fig, axs = plt.subplots(nrows=15, ncols=15, figsize=(15, 15))
        axs = axs.ravel()

        if misclassified:
            x_test = x_test[mc_idxs]
            y_test = y_test[mc_idxs]
            y_pred = y_pred[mc_idxs]
            idxs = mc_idxs
        else:
            idxs = np.arange(start=0, stop=len(y_pred), step=1)

        for i, x, y, y_p, ax in zip(idxs, x_test, y_test, y_pred, axs):
            ax.imshow(x, cmap='seismic', clim=(-1, 1))
            ax.set_title('i: {}, pred: {}, true: {}'.format(i, np.argmax(y_p), np.argmax(y)), fontsize=6)

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig('{}_examples_indices'.format(path))
    else:
        compare_classes(methods, model, x_test, y_test, y_pred, cls1, cls2, path, indices, misclassified=misclassified)
        compare_classes(methods, model, x_test, y_test, y_pred, cls1, cls2, path, indices, thresholded=True, misclassified=misclassified)

    # Clear session
    clear_session()
