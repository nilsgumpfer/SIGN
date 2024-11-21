import os

import pandas as pd

from experiments.evaluate import run_evaluate_bulk
from utils.env import set_visible_gpu

set_visible_gpu(2)

from experiments.bulk_mnist import run_bulk_mnist
from utils.mnist import load_and_prepare_data, load_or_train_model
from utils.training import enable_reproducibility
from matplotlib import rcParams

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = "\\usepackage{amssymb}\n \\usepackage{amsmath}"
rcParams['savefig.format'] = 'pdf'
rcParams['figure.dpi'] = 1000

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

def generate_plot():
    for root, dirs, files in os.walk('tables'):
        for f in files:
            if f.endswith('.xlsx'):
                df = pd.read_excel('{}/{}'.format(root, f))
                print(df.columns)
                # TODO: GO ON HERE

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