import numpy as np

from utils.innv import aggregate_and_normalize_relevancemap_rgb
from scipy.stats import spearmanr as scipyspearmanr


def generate_position_tuples(h):
    values = []
    positions = []

    for px in range(h.shape[0]):
        for py in range(h.shape[1]):
            values.append(h[px][py])
            positions.append((px, py))

    position_tuples = list(zip(values, positions))
    position_tuples_sorted = sorted(position_tuples, reverse=True)

    return position_tuples_sorted


def generate_replacement_values(n, intervals, seed=1):
    np.random.seed(seed)

    channel_values = []

    for low, high in intervals:
        uniform_values = np.random.uniform(low=low, high=high, size=n)
        channel_values.append(uniform_values)

    replacement_values = np.column_stack(channel_values)

    return replacement_values


def derive_channel_based_intervals(x):
    intervals = []
    for c in range(x.shape[-1]):
        v = x[:, :, c].ravel()
        intervals.append((np.min(v), np.max(v)))

    return intervals


def generate_perturbed_heatmaps(H, position_tuples_sorted, num=None):
    # Copy input
    H_tmp = np.array(H)

    # Derive number of positions / pixels
    npx = len(position_tuples_sorted)

    # Initialize placeholder for perturbed heatmaps
    perturbed_heatmaps = [np.array(H_tmp)]

    # Derive stop points
    if num is None:
        n = npx
    else:
        n = num
    stop_points = np.linspace(start=0, stop=npx - 1, num=n, dtype=int)

    # Replace pixel values based on drawn replacement values
    for i, (rv, (px, py)) in enumerate(position_tuples_sorted):
        H_tmp[px][py] = 0

        if i in stop_points:
            perturbed_heatmaps.append(np.array(H_tmp))

    return np.array(perturbed_heatmaps)


def generate_perturbed_inputs(x, position_tuples_sorted, num=None):
    # Copy input
    x_tmp = np.array(x)

    # Derive number of positions / pixels
    npx = len(position_tuples_sorted)

    # Derive intervals
    intervals = derive_channel_based_intervals(x)

    # Draw random replacement values from uniform distribution
    replacement_values = generate_replacement_values(npx, intervals)

    # Initialize placeholder for perturbed inputs
    perturbed_inputs = [np.array(x_tmp)]

    # Derive stop points
    if num is None:
        n = npx
    else:
        n = num
    stop_points = np.linspace(start=0, stop=npx - 1, num=n, dtype=int)

    # Replace pixel values based on drawn replacement values
    for i, (rv, (px, py)) in enumerate(position_tuples_sorted):
        x_tmp[px][py] = replacement_values[i]

        if i in stop_points:
            perturbed_inputs.append(np.array(x_tmp))

    return np.array(perturbed_inputs)


def calculate_predictions(perturbed_inputs, model_softmax, batch_size=None, idx=None):
    if batch_size is None:
        predictions = model_softmax.predict_on_batch(perturbed_inputs)

    else:
        batches = np.array_split(perturbed_inputs, int(len(perturbed_inputs) / batch_size))
        predictions_batches = []

        for batch in batches:
            predictions_batch = model_softmax.predict_on_batch(batch)
            predictions_batches.append(predictions_batch)

        predictions = np.vstack(predictions_batches)

    if idx is None:
        idx = int(np.argmax(predictions[0]))

    return predictions[:, idx]


def calculate_morf_curve(x, R, model_softmax, morfnum, batch_size=None, idx=None):
    # Aggregate relevancemap on pixel-level
    H = np.array(aggregate_and_normalize_relevancemap_rgb(R))

    # Generate sorted list of pixel positions based on their relevance (highest first)
    position_tuples_sorted = generate_position_tuples(H)

    # Generate perturbed inputs
    perturbed_inputs = generate_perturbed_inputs(x, position_tuples_sorted, num=morfnum)

    # Calculate predictions
    prediction_values = calculate_predictions(perturbed_inputs, model_softmax, batch_size, idx=idx)

    return prediction_values


def calculate_mean_aoc_curve(morf_curves, return_mean_morf_curve=False):
    aocs = []

    # Calculate mean MoRF curve over all MoRF curves
    mean_curve = np.mean(morf_curves, axis=0)

    for i in range(len(mean_curve) - 1):
        aocs.append(calculate_aoc(mean_curve[0:i + 2]))

    if return_mean_morf_curve:
        return np.array(aocs), mean_curve
    else:
        return np.array(aocs)


def calculate_mean_aoc_from_curves(morf_curves):
    aocs = np.zeros(len(morf_curves))

    for i, mc in enumerate(morf_curves):
        aocs[i] = calculate_aoc(mc)

    return np.mean(aocs), np.std(aocs)


def load_morf_curves(dataset_id, model_id, method):
    # Derive path
    path_results = 'results/{}_{}'.format(dataset_id, model_id)

    # Load file
    morf_curves = np.load('{}/morfcurves_{}.npy'.format(path_results, method))

    return morf_curves


def load_metrics(dataset_id, model_id, method, metric):
    # Derive path
    path_results = 'results/{}_{}'.format(dataset_id, model_id)

    # Load file
    values = np.load('{}/{}values_{}.npy'.format(path_results, metric, method))

    return values


def calculate_aoc(y):
    aoc = 100 - np.trapz(y, x=np.linspace(start=1, stop=100, num=len(y)))

    return aoc / 100.0


def input_relevance_correlation(x, h, return_values=False, variant='pearson'):
    # Derive values
    xvalues = np.ravel(np.abs(x))
    hvalues = np.ravel(np.abs(h))

    if variant == 'pearson':
        # Calculate pearson's r correlation coefficient
        r = pearsonr(xvalues, hvalues)
    elif variant == 'spearman':
        # Calculate spearman's r correlation coefficient
        r = spearmanr(xvalues, hvalues)
    else:
        raise ValueError('Unknown variant for calulcation of value correlation: "{}"'.format(variant))

    if return_values:
        return xvalues, hvalues, r
    else:
        return r


def spearmanr(a, b):
    r, p = scipyspearmanr(a, b, nan_policy='omit')

    return r


def pearsonr(a, b):
    amean = np.mean(a)
    bmean = np.mean(b)

    am = a - amean
    bm = b - bmean

    numerator = np.sum(am * bm)
    tmp1 = np.sqrt(np.sum(am**2.0))
    tmp2 = np.sqrt(np.sum(bm**2.0))

    denominator = tmp1 * tmp2

    r = numerator / denominator

    return r
