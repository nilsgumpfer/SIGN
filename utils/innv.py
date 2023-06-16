import numpy as np
from methods import innvestigate


def calculate_explanation_innvestigate(model, x, method='lrp.epsilon', neuron_selection=None, batchmode=False, **kwargs):
    analyzer = innvestigate.create_analyzer(method, model, **kwargs)

    if neuron_selection is None:
        neuron_selection = 'max_activation'

    if not batchmode:
        ex = analyzer.analyze(X=[x], neuron_selection=neuron_selection, **kwargs)
        expl = ex[list(ex.keys())[0]][0]

        return np.asarray(expl)
    else:
        ex = analyzer.analyze(X=x, neuron_selection=neuron_selection, **kwargs)
        expl = ex[list(ex.keys())[0]]

        return np.asarray(expl)


def aggregate_and_normalize_relevancemap_rgb(R):
    # Aggregate along color channels and normalize to [-1, 1]
    a = R.sum(axis=2)
    a = normalize_heatmap(a)

    return a


def normalize_heatmap(H):
    # Normalize to [-1, 1]
    a = H / np.max(np.abs(H))

    a = np.nan_to_num(a, nan=0)

    return a
