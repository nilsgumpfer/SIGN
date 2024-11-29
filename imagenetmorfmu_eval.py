import os
from symbol import factor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tensorflow.python.layers.core import dense

from experiments.evaluate import run_evaluate_bulk
from utils.env import set_visible_gpu

set_visible_gpu(2)

from experiments.bulk_mnist import run_bulk_mnist
from utils.mnist import load_and_prepare_data, load_or_train_model
from utils.training import enable_reproducibility
from matplotlib import rcParams

# rcParams['text.usetex'] = True
# rcParams['font.size'] = 14
# rcParams['text.latex.preamble'] = "\\usepackage{amssymb}\n \\usepackage{amsmath}"
# rcParams['savefig.format'] = 'pdf'
# rcParams['figure.dpi'] = 1000

methods = ['gradient_x_sign_mu_128',
           'gradient_x_sign_mu_96',
           'gradient_x_sign_mu_64',
           'gradient_x_sign_mu_32',
           'gradient_x_sign_mu_0',
           'gradient_x_sign_mu_neg_32',
           'gradient_x_sign_mu_neg_64',
           'gradient_x_sign_mu_neg_96',
           'gradient_x_sign_mu_neg_128']


def parse_float(s):
    if s.startswith('neg'):
        s = s.replace('neg_', '')
        factor = -1.0
    else:
        factor = 1.0

    return float(s.replace('_', '.')) * factor


def generate_plot():
        x = []
        y = []
        df = pd.read_excel('tables/ILSVRC2012val_VGG16ILSVRC/table_results_g1.xlsx', engine='openpyxl', skiprows=1)

        for m, aoc in zip(df[df.columns[0]].values[1:], df['mean'].values[1:]):
            begin = 'gradient_x_sign_mu_'
            if m.startswith(begin):
                mu = parse_float(m.replace(begin, ''))
                x.append(mu)
                y.append(aoc)

        title = 'ILSVRC2012val_VGG16ILSVRC'

        x = np.array(x)
        y = np.nan_to_num(np.array(y))
        plt.scatter(x, y)
        plt.scatter(x[y == np.max(y)], y[y == np.max(y)], c='r')
        plt.ylim((0.85, max(y) + 0.01))
        plt.xlim((-141 - 0.02, 141))
        plt.title(title)
        plt.tight_layout()
        plt.savefig('plots/mu_analysis/{}.pdf'.format(title))
        plt.close()


run_evaluate_bulk(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC', methods=methods, legend_inside=False)
generate_plot()