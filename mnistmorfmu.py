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

methods = ['random_uniform',
            'gradient',
            'gradient_x_input',
            'gradient_x_sign_mu_1',
            'gradient_x_sign_mu_0_9',
            'gradient_x_sign_mu_0_8',
            'gradient_x_sign_mu_0_7',
            'gradient_x_sign_mu_0_6',
            'gradient_x_sign_mu_0_5',
            'gradient_x_sign_mu_0_4',
            'gradient_x_sign_mu_0_3',
            'gradient_x_sign_mu_0_2',
            'gradient_x_sign_mu_0_1',
            'gradient_x_sign_mu_0',
            'gradient_x_sign_mu_neg_0_1',
            'gradient_x_sign_mu_neg_0_2',
            'gradient_x_sign_mu_neg_0_3',
            'gradient_x_sign_mu_neg_0_4',
            'gradient_x_sign_mu_neg_0_5',
            'gradient_x_sign_mu_neg_0_6',
            'gradient_x_sign_mu_neg_0_7',
            'gradient_x_sign_mu_neg_0_8',
            'gradient_x_sign_mu_neg_0_9',
            'gradient_x_sign_mu_neg_1']

def train_MNIST(variant, net, epochs=5, random_state=0, inverted=False):
    # Reproducibility setup
    enable_reproducibility(random_state)

    # Prepare labels and paths
    invlbl = {True: 'INV', False: ''}
    model_id = '{}MNIST{}{}'.format(net, variant, invlbl[inverted])
    modelpath = 'data/models/{}.h5'.format(model_id)

    # Load and prepare data
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data(variant, inverted)

    # Train model
    print(model_id)
    load_or_train_model(modelpath, net, x_train, y_train, x_test, y_test, epochs)


def parse_float(s):
    if s.startswith('neg'):
        s = s.replace('neg_', '')
        factor = -1.0
    else:
        factor = 1.0

    return float(s.replace('_', '.')) * factor


def generate_plot():
    for root, dirs, files in os.walk('tables'):
        for f in files:
            if f.endswith('.xlsx'):
                x = []
                y = []
                df = pd.read_excel('{}/{}'.format(root, f), engine='openpyxl', skiprows=1)

                for m, aoc in zip(df[df.columns[0]].values[1:], df['mean'].values[1:]):
                    begin = 'gradient_x_sign_mu_'
                    if m.startswith(begin):
                        mu = parse_float(m.replace(begin, ''))
                        x.append(mu)
                        y.append(aoc)

                title = root.replace('tables/', '')
                variant = title.replace('MNIST_DENSEMNIST', '')
                if variant.endswith('INV'):
                    variant = variant.replace('INV', '')
                    inverted = True
                else:
                    inverted = False

                # Load and prepare data
                print('Loading', variant, inverted)
                (_, _), (x_test, _) = load_and_prepare_data(variant, inverted)
                data = np.ravel(x_test)

                # for q in np.arange(start=0, stop=1, step=0.05):
                #     d = np.quantile(np.ravel(x_test), q)
                #     print(q, d)

                plt.hist(data, bins=50, density=True, alpha=0.6, color='blue', edgecolor='black')

                # d = np.quantile(np.ravel(x_test), 0.125)
                # plt.axvline(x=d, color='r')
                plt.scatter(x, y)
                # plt.ylim((0.85, max(y)+0.02))
                plt.title(title)
                plt.tight_layout()
                plt.savefig('plots/mu_analysis/{}.pdf'.format(title))
                plt.close()
                del x_test


# Train MNIST models
# train_MNIST(variant='01', net='DENSE', inverted=True)
# train_MNIST(variant='01', net='DENSE', inverted=False)
# train_MNIST(variant='11', net='DENSE', inverted=True)
# train_MNIST(variant='11', net='DENSE', inverted=False)
# train_MNIST(variant='10', net='DENSE', inverted=True)
# train_MNIST(variant='10', net='DENSE', inverted=False)

# Run MNIST experiments based on previously trained models
# run_bulk_mnist(methods=methods, variant='11', net='DENSE', inverted=False, calc_pcc=False, calc_scc=False)
# run_bulk_mnist(methods=methods, variant='11', net='DENSE', inverted=True, calc_pcc=False, calc_scc=False)
# run_bulk_mnist(methods=methods, variant='10', net='DENSE', inverted=False, calc_pcc=False, calc_scc=False)
# run_bulk_mnist(methods=methods, variant='10', net='DENSE', inverted=True, calc_pcc=False, calc_scc=False)
# run_bulk_mnist(methods=methods, variant='01', net='DENSE', inverted=False, calc_pcc=False, calc_scc=False)
# run_bulk_mnist(methods=methods, variant='01', net='DENSE', inverted=True, calc_pcc=False, calc_scc=False)

# Evaluate experiments
# run_evaluate_bulk(dataset_id='MNIST', model_id='DENSEMNIST11', group_name='D11', methods=methods, noplot=True)
# run_evaluate_bulk(dataset_id='MNIST', model_id='DENSEMNIST11INV', group_name='D11INV', methods=methods, noplot=True)
# run_evaluate_bulk(dataset_id='MNIST', model_id='DENSEMNIST01', group_name='D01', methods=methods, noplot=True)
# run_evaluate_bulk(dataset_id='MNIST', model_id='DENSEMNIST01INV', group_name='D01INV', methods=methods, noplot=True)
# run_evaluate_bulk(dataset_id='MNIST', model_id='DENSEMNIST10', group_name='D10', methods=methods, noplot=True)
# run_evaluate_bulk(dataset_id='MNIST', model_id='DENSEMNIST10INV', group_name='D10INV', methods=methods, noplot=True)

generate_plot()