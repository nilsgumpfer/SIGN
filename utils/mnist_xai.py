import numpy as np
from matplotlib import pyplot as plt

from methods.wrappers import calculate_relevancemaps
from utils.misc import pretty_method_name


def compare_classes(methods, model, x, y, y_pred, cls1, cls2, path, indices, cmap='seismic', thresholded=False, misclassified=False):
    X = x[indices]
    n = len(indices)
    print(indices)

    # Derive levels for heatmap contours
    v = np.ravel(x)
    mn, mx = np.min(v), np.max(v)
    levels = [(mn + mx) / 2.0]

    # Prepare plot
    fig, axs = plt.subplots(nrows=n, ncols=len(methods) + 1, figsize=((len(methods) + 1) * 5, n * 5))

    # Plot inputs
    for i, x in enumerate(X):
        axs[i][0].imshow(x, cmap='seismic', clim=(-1, 1))

    axs[0][0].set_title('Input', fontsize=36, rotation=-22.5, horizontalalignment="right", verticalalignment="bottom")

    # Plot heatmaps method-wise
    for c, m in enumerate(methods):
        Rs = calculate_relevancemaps(m, np.array(X), model, neuron_selection=cls2)

        axs[0][c + 1].set_title(pretty_method_name(m, wo_params=True), fontsize=36, rotation=-22.5, horizontalalignment="right", verticalalignment="bottom")

        for r in range(n):
            Hn = Rs[r] / np.max(np.abs(np.ravel(Rs[r])))

            if thresholded:
                Hn[Hn < 0] = -0.5
                Hn[Hn > 0] = 0.5

            axs[r][c + 1].imshow(Hn, cmap=cmap, clim=(-1, 1))
            axs[r][c + 1].contour(np.squeeze(X[r], axis=2), levels=levels, linewidths=[5.0], colors='k', linestyles='-')

    for ax in np.ravel(axs):
        ax.set_xticks([])
        ax.set_yticks([])

    for r, i in enumerate(indices):
        y_c = np.argmax(y[i])
        y_c_p = np.argmax(y_pred[i])
        axs[r][0].set_xlabel('true: {}, pred: {}, prob: {:.2f}'.format(y_c, y_c_p, y_pred[i][y_c_p]), fontsize=32)

    if not thresholded and not misclassified:
        suffix = ''
    else:
        suffix = '_'
        if thresholded:
            suffix += 'thresholded'
        if misclassified:
            suffix += 'misclassified'

    plt.savefig('{}_{}_vs_{}{}'.format(path, cls1, cls2, suffix), orientation="landscape", bbox_inches="tight")

    plt.close()
