from utils.env import set_max_threads, set_visible_gpu

set_max_threads(5)
set_visible_gpu(0)

from matplotlib import rcParams
rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = "\\usepackage{amssymb}\n \\usepackage{amsmath}"
rcParams['savefig.format'] = 'pdf'

from experiments.correlation import run_dataset_n_CC_map

from experiments.singleMoRF_mnist import run_single_MoRF_mnist
from experiments.bulk_mnist import run_bulk_mnist
from experiments.bulk import run_bulk
from experiments.evaluate import run_evaluate_bulk, run_evaluate_bulk_multiple, run_aoc_corr_plot
from experiments.single import run_single, run_single_explain_classes
from experiments.singleMoRF import run_single_MoRF
from utils.config import get_methods, get_modelids
from utils.image import plot_single, plot_crop_zoom, plot_combine_row
from experiments.multiple import run_multiple
from experiments.simple_mnist import run_simple_MNIST

# Run experiments
run_bulk(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC', methods=get_methods(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC'))
run_bulk(dataset_id='MITPLACES365val', model_id='VGG16MITPL365', methods=get_methods(dataset_id='MITPLACES365val', model_id='VGG16MITPL365'))

# Generate result tables
run_evaluate_bulk(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC', group_name='basic', methods=get_methods(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC', variant='basic'), noplot=True)
run_evaluate_bulk(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC', group_name='LRPall', methods=get_methods(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC', variant='LRPall'), noplot=True)
run_evaluate_bulk(dataset_id='MITPLACES365val', model_id='VGG16MITPL365', group_name='all', methods=get_methods(dataset_id='MITPLACES365val', model_id='VGG16MITPL365'), noplot=True)

# Generate MoRF curve plots
run_evaluate_bulk(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC', group_name='plotimagenet', methods=get_methods(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC', variant='morf'), legend_inside=False)
run_evaluate_bulk(dataset_id='MITPLACES365val', model_id='VGG16MITPL365', group_name='plotplaces', methods=get_methods(dataset_id='MITPLACES365val', model_id='VGG16MITPL365', variant='morf'), legend_inside=False)

# Generate figure with multiple examples (VGG16ILSVRC)
run_multiple(dataset_id='flickr', model_id='VGG16ILSVRC', group_name='two', methods=get_methods(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC', variant='fig'), filenames=['27594777915_9ed2bc78f9_o.jpg', '11425971435_3cedb1ac05_c.jpg'])
run_multiple(dataset_id='flickr', model_id='VGG16ILSVRC', group_name='results', methods=get_methods(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC', variant='fig'), filenames=['20998580825_d1ed50e38e_o.jpg', '605505232_e93d1a976f_o.jpg', '234775821_ae4ebf6f2a_o.jpg', '27594777915_9ed2bc78f9_o.jpg'])

# Misclassified examples (VGG16ILSVRC)
run_multiple(dataset_id='flickr', model_id='VGG16ILSVRC', group_name='misclassified', methods=get_methods(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC', variant='fig'), filenames=['7303748070_11b8d8f04e_o.jpg', '24377489488_62762c9e79_o.jpg', '2832792280_a9b2d78a6b_o.jpg'])

# Generate figure with single example with multi-class explanations (VGG16ILSVRC)
run_single_explain_classes(dataset_id='flickr', model_id='VGG16ILSVRC', class_idxs=[386, 609], filename='8740058525_5f9b8868e9_o.jpg', methods=get_methods(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC', variant='fig'))

# Generate figure with multiple examples (VGG16MITPL365)
run_multiple(dataset_id='flickr', model_id='VGG16MITPL365', group_name='results', methods=get_methods(dataset_id='MITPLACES365val', model_id='VGG16MITPL365', variant='fig'), filenames=['3842551283_c289a49691_o.jpg', '41586583205_0ed4325345_o.jpg', '14941412_371ac3f038_o.jpg', '15519846401_582d9bcbac_o.jpg'])

# MoRF plots VGG16ILSVRC
run_single_MoRF('gradient_x_input', 'flickr', 'VGG16ILSVRC', '447888592_7e64c84851_o.jpg', morfnum=250)
run_single_MoRF('gradient_x_sign_mu_0', 'flickr', 'VGG16ILSVRC', '447888592_7e64c84851_o.jpg', morfnum=250)
run_single_MoRF('lrpz_epsilon_0_25_std_x', 'flickr', 'VGG16ILSVRC', '447888592_7e64c84851_o.jpg', morfnum=250)
run_single_MoRF('lrpsign_epsilon_0_25_std_x', 'flickr', 'VGG16ILSVRC', '447888592_7e64c84851_o.jpg', morfnum=250)
run_single_MoRF('gradient_x_input', 'flickr', 'VGG16ILSVRC', '41535389324_47768da87c_c.jpg', morfnum=250)
run_single_MoRF('gradient_x_sign_mu_0', 'flickr', 'VGG16ILSVRC', '41535389324_47768da87c_c.jpg', morfnum=250)
run_single_MoRF('lrpz_epsilon_0_25_std_x', 'flickr', 'VGG16ILSVRC', '41535389324_47768da87c_c.jpg', morfnum=250)
run_single_MoRF('lrpsign_epsilon_0_25_std_x', 'flickr', 'VGG16ILSVRC', '41535389324_47768da87c_c.jpg', morfnum=250)

# MoRF plots VGG16MITPL365
run_single_MoRF('gradient_x_input', 'flickr', 'VGG16MITPL365', '6126018690_05633db4f9_o.jpg', morfnum=250)
run_single_MoRF('gradient_x_sign_mu_0', 'flickr', 'VGG16MITPL365', '6126018690_05633db4f9_o.jpg', morfnum=250)
run_single_MoRF('lrpz_epsilon_0_25_std_x', 'flickr', 'VGG16MITPL365', '6126018690_05633db4f9_o.jpg', morfnum=250)
run_single_MoRF('lrpsign_epsilon_0_25_std_x', 'flickr', 'VGG16MITPL365', '6126018690_05633db4f9_o.jpg', morfnum=250)
run_single_MoRF('gradient_x_input', 'flickr', 'VGG16MITPL365', '9527376792_66fd86567e_o.jpg', morfnum=250)
run_single_MoRF('gradient_x_sign_mu_0', 'flickr', 'VGG16MITPL365', '9527376792_66fd86567e_o.jpg', morfnum=250)
run_single_MoRF('lrpz_epsilon_0_25_std_x', 'flickr', 'VGG16MITPL365', '9527376792_66fd86567e_o.jpg', morfnum=250)
run_single_MoRF('lrpsign_epsilon_0_25_std_x', 'flickr', 'VGG16MITPL365', '9527376792_66fd86567e_o.jpg', morfnum=250)

# Train MNIST models, run MNIST plots
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST11')[1:], variant='11', net='DENSE', epochs=5, inverted=False, cls1=3, cls2=3, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST11')[1:], variant='11', net='DENSE', epochs=5, inverted=False, cls1=3, cls2=8, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST11INV')[1:], variant='11', net='DENSE', epochs=5, inverted=True, cls1=3, cls2=3, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST11INV')[1:], variant='11', net='DENSE', epochs=5, inverted=True, cls1=3, cls2=8, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST10')[1:], variant='10', net='DENSE', epochs=5, inverted=False, cls1=3, cls2=3, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST10')[1:], variant='10', net='DENSE', epochs=5, inverted=False, cls1=3, cls2=8, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST10INV')[1:], variant='10', net='DENSE', epochs=5, inverted=True, cls1=3, cls2=3, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST10INV')[1:], variant='10', net='DENSE', epochs=5, inverted=True, cls1=3, cls2=8, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST01')[1:], variant='01', net='DENSE', epochs=5, inverted=False, cls1=3, cls2=3, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST01')[1:], variant='01', net='DENSE', epochs=5, inverted=False, cls1=3, cls2=8, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST01INV')[1:], variant='01', net='DENSE', epochs=5, inverted=True, cls1=3, cls2=3, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST01INV')[1:], variant='01', net='DENSE', epochs=5, inverted=True, cls1=3, cls2=8, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'CNNMNIST11')[1:], variant='11', net='CNN', epochs=3, inverted=False, cls1=3, cls2=3, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'CNNMNIST11')[1:], variant='11', net='CNN', epochs=3, inverted=False, cls1=3, cls2=8, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'CNNMNIST11INV')[1:], variant='11', net='CNN', epochs=3, inverted=True, cls1=3, cls2=3, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'CNNMNIST11INV')[1:], variant='11', net='CNN', epochs=3, inverted=True, cls1=3, cls2=8, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'CNNMNIST10')[1:], variant='10', net='CNN', epochs=3, inverted=False, cls1=3, cls2=3, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'CNNMNIST10')[1:], variant='10', net='CNN', epochs=3, inverted=False, cls1=3, cls2=8, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'CNNMNIST10INV')[1:], variant='10', net='CNN', epochs=3, inverted=True, cls1=3, cls2=3, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'CNNMNIST10INV')[1:], variant='10', net='CNN', epochs=3, inverted=True, cls1=3, cls2=8, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'CNNMNIST01')[1:], variant='01', net='CNN', epochs=3, inverted=False, cls1=3, cls2=3, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'CNNMNIST01')[1:], variant='01', net='CNN', epochs=3, inverted=False, cls1=3, cls2=8, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'CNNMNIST01INV')[1:], variant='01', net='CNN', epochs=3, inverted=True, cls1=3, cls2=3, indices=[231, 110, 10])
run_simple_MNIST(methods=get_methods('MNIST', 'CNNMNIST01INV')[1:], variant='01', net='CNN', epochs=3, inverted=True, cls1=3, cls2=8, indices=[231, 110, 10])

# Plot two misclassified examples (MNIST)
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST10INV')[1:], variant='10', net='DENSE', epochs=5, inverted=True, cls1='3,8', cls2=3, indices=[2004, 6046], misclassified=True)
run_simple_MNIST(methods=get_methods('MNIST', 'DENSEMNIST10INV')[1:], variant='10', net='DENSE', epochs=5, inverted=True, cls1='3,8', cls2=8, indices=[2004, 6046], misclassified=True)

# Run MNIST experiments based on previously trained models
run_bulk_mnist(methods=get_methods('MNIST', 'DENSEMNIST11'), variant='11', net='DENSE', inverted=False)
run_bulk_mnist(methods=get_methods('MNIST', 'DENSEMNIST11INV'), variant='11', net='DENSE', inverted=True)
run_bulk_mnist(methods=get_methods('MNIST', 'DENSEMNIST10'), variant='10', net='DENSE', inverted=False)
run_bulk_mnist(methods=get_methods('MNIST', 'DENSEMNIST10INV'), variant='10', net='DENSE', inverted=True)
run_bulk_mnist(methods=get_methods('MNIST', 'DENSEMNIST01'), variant='01', net='DENSE', inverted=False)
run_bulk_mnist(methods=get_methods('MNIST', 'DENSEMNIST01INV'), variant='01', net='DENSE', inverted=True)
run_bulk_mnist(methods=get_methods('MNIST', 'CNNMNIST11'), variant='11', net='CNN', inverted=False)
run_bulk_mnist(methods=get_methods('MNIST', 'CNNMNIST11INV'), variant='11', net='CNN', inverted=True)
run_bulk_mnist(methods=get_methods('MNIST', 'CNNMNIST10'), variant='10', net='CNN', inverted=False)
run_bulk_mnist(methods=get_methods('MNIST', 'CNNMNIST10INV'), variant='10', net='CNN', inverted=True)
run_bulk_mnist(methods=get_methods('MNIST', 'CNNMNIST01'), variant='01', net='CNN', inverted=False)
run_bulk_mnist(methods=get_methods('MNIST', 'CNNMNIST01INV'), variant='01', net='CNN', inverted=True)

# Evaluate MNIST experiments, generate plots
run_evaluate_bulk(dataset_id='MNIST', model_id='DENSEMNIST11', group_name='D11', methods=get_methods('MNIST', 'DENSEMNIST11'))
run_evaluate_bulk(dataset_id='MNIST', model_id='DENSEMNIST11INV', group_name='D11INV', methods=get_methods('MNIST', 'DENSEMNIST11INV'))
run_evaluate_bulk(dataset_id='MNIST', model_id='DENSEMNIST01', group_name='D01', methods=get_methods('MNIST', 'DENSEMNIST01'))
run_evaluate_bulk(dataset_id='MNIST', model_id='DENSEMNIST01INV', group_name='D01INV', methods=get_methods('MNIST', 'DENSEMNIST01INV'))
run_evaluate_bulk(dataset_id='MNIST', model_id='DENSEMNIST10', group_name='D10', methods=get_methods('MNIST', 'DENSEMNIST10'))
run_evaluate_bulk(dataset_id='MNIST', model_id='DENSEMNIST10INV', group_name='D10INV', methods=get_methods('MNIST', 'DENSEMNIST10INV'))
run_evaluate_bulk(dataset_id='MNIST', model_id='CNNMNIST11', group_name='C11', methods=get_methods('MNIST', 'CNNMNIST11'))
run_evaluate_bulk(dataset_id='MNIST', model_id='CNNMNIST11INV', group_name='C11INV', methods=get_methods('MNIST', 'CNNMNIST11INV'))
run_evaluate_bulk(dataset_id='MNIST', model_id='CNNMNIST01', group_name='C01', methods=get_methods('MNIST', 'CNNMNIST01'))
run_evaluate_bulk(dataset_id='MNIST', model_id='CNNMNIST01INV', group_name='C01INV', methods=get_methods('MNIST', 'CNNMNIST01INV'))
run_evaluate_bulk(dataset_id='MNIST', model_id='CNNMNIST10', group_name='C10', methods=get_methods('MNIST', 'CNNMNIST10'))
run_evaluate_bulk(dataset_id='MNIST', model_id='CNNMNIST10INV', group_name='C10INV', methods=get_methods('MNIST', 'CNNMNIST10INV'))

# Evaluate multiple MNIST models, generate combined table for each metric
run_evaluate_bulk_multiple(dataset_id='MNIST', model_ids=get_modelids('MNIST'), group_name='all', methods=get_methods('MNIST', 'GENERAL'), metric='pcc', dround=2, aggr=['min', 'max', 'mean', 'std'], colgroups={'net': ['CNN', 'DENSE'], 'variant': ['01', '11', '10'], 'INV': [False, True]}, suffix_merge={r'_mu_repl': [r'_mu_neg_0_5', r'_mu_0', r'_mu_0_5']})
run_evaluate_bulk_multiple(dataset_id='MNIST', model_ids=get_modelids('MNIST'), group_name='all', methods=get_methods('MNIST', 'GENERAL'), metric='scc', dround=2, aggr=['min', 'max', 'mean', 'std'], colgroups={'net': ['CNN', 'DENSE'], 'variant': ['01', '11', '10'], 'INV': [False, True]}, suffix_merge={r'_mu_repl': [r'_mu_neg_0_5', r'_mu_0', r'_mu_0_5']})
run_evaluate_bulk_multiple(dataset_id='MNIST', model_ids=get_modelids('MNIST'), group_name='all', methods=get_methods('MNIST', 'GENERAL'), metric='aoc', dround=4, aggr=['min', 'max', 'mean', 'std'], colgroups={'net': ['CNN', 'DENSE'], 'variant': ['01', '11', '10'], 'INV': [False, True]}, suffix_merge={r'_mu_repl': [r'_mu_neg_0_5', r'_mu_0', r'_mu_0_5']})

# MNIST MoRF plots
run_single_MoRF_mnist(method='gradient_x_input', variant='01', net='CNN', inverted=False, ind=19, morfnum=250)
run_single_MoRF_mnist(method='gradient_x_sign_mu_0_5', variant='01', net='CNN', inverted=False, ind=19, morfnum=250)
run_single_MoRF_mnist(method='gradient_x_input', variant='01', net='CNN', inverted=False, ind=1, morfnum=250)
run_single_MoRF_mnist(method='gradient_x_sign_mu_0_5', variant='01', net='CNN', inverted=False, ind=1, morfnum=250)
run_single_MoRF_mnist(method='gradient_x_input', variant='01', net='CNN', inverted=False, ind=5, morfnum=250)
run_single_MoRF_mnist(method='gradient_x_sign_mu_0_5', variant='01', net='CNN', inverted=False, ind=5, morfnum=250)
run_single_MoRF_mnist(method='gradient_x_input', variant='10', net='CNN', inverted=False, ind=5, morfnum=250)
run_single_MoRF_mnist(method='gradient_x_sign_mu_neg_0_5', variant='10', net='CNN', inverted=False, ind=5, morfnum=250)
run_single_MoRF_mnist(method='gradient_x_input', variant='11', net='CNN', inverted=False, ind=1, morfnum=250)
run_single_MoRF_mnist(method='gradient_x_sign_mu_0', variant='11', net='CNN', inverted=False, ind=1, morfnum=250)
run_single_MoRF_mnist(method='gradient_x_input', variant='01', net='DENSE', inverted=False, ind=1, morfnum=250)
run_single_MoRF_mnist(method='gradient_x_sign_mu_0_5', variant='01', net='DENSE', inverted=False, ind=1, morfnum=250)
run_single_MoRF_mnist(method='gradient_x_input', variant='10', net='DENSE', inverted=False, ind=19, morfnum=250)
run_single_MoRF_mnist(method='gradient_x_sign_mu_neg_0_5', variant='10', net='DENSE', inverted=False, ind=19, morfnum=250)


# Run further plots with different DPI requirements

# AOC vs PCC plots
run_aoc_corr_plot([{'dataset_id': 'MNIST', 'model_ids': get_modelids('MNIST'), 'methods': get_methods('MNIST', 'GENERAL')},
                   {'dataset_id': 'ILSVRC2012val', 'model_ids': ['VGG16ILSVRC'], 'methods': get_methods(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC')},
                   {'dataset_id': 'MITPLACES365val', 'model_ids': ['VGG16MITPL365'], 'methods': get_methods(dataset_id='MITPLACES365val', model_id='VGG16MITPL365')}],
                  metric='pcc',
                  show_labels=False)
run_aoc_corr_plot([{'dataset_id': 'MNIST', 'model_ids': get_modelids('MNIST'), 'methods': get_methods('MNIST', 'GENERAL')},
                   {'dataset_id': 'ILSVRC2012val', 'model_ids': ['VGG16ILSVRC'], 'methods': get_methods(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC')},
                   {'dataset_id': 'MITPLACES365val', 'model_ids': ['VGG16MITPL365'], 'methods': get_methods(dataset_id='MITPLACES365val', model_id='VGG16MITPL365')}],
                  metric='pcc',
                  show_labels=True)

# AOC vs SCC plots
run_aoc_corr_plot([{'dataset_id': 'MNIST', 'model_ids': get_modelids('MNIST'), 'methods': get_methods('MNIST', 'GENERAL')},
                   {'dataset_id': 'ILSVRC2012val', 'model_ids': ['VGG16ILSVRC'], 'methods': get_methods(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC')},
                   {'dataset_id': 'MITPLACES365val', 'model_ids': ['VGG16MITPL365'], 'methods': get_methods(dataset_id='MITPLACES365val', model_id='VGG16MITPL365')}],
                  metric='scc',
                  show_labels=False)
run_aoc_corr_plot([{'dataset_id': 'MNIST', 'model_ids': get_modelids('MNIST'), 'methods': get_methods('MNIST', 'GENERAL')},
                   {'dataset_id': 'ILSVRC2012val', 'model_ids': ['VGG16ILSVRC'], 'methods': get_methods(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC')},
                   {'dataset_id': 'MITPLACES365val', 'model_ids': ['VGG16MITPL365'], 'methods': get_methods(dataset_id='MITPLACES365val', model_id='VGG16MITPL365')}],
                  metric='scc',
                  show_labels=True)

# CC plots VGG16ILSVRC
rcParams['figure.dpi'] = 500
run_dataset_n_CC_map('gradient_x_input', 'ILSVRC2012val', 'VGG16ILSVRC', n=10000)
run_dataset_n_CC_map('gradient_x_sign_mu_0', 'ILSVRC2012val', 'VGG16ILSVRC', n=10000)
run_dataset_n_CC_map('lrpz_epsilon_0_25_std_x', 'ILSVRC2012val', 'VGG16ILSVRC', n=10000)
run_dataset_n_CC_map('lrpsign_epsilon_0_25_std_x_mu_0', 'ILSVRC2012val', 'VGG16ILSVRC', n=10000)

# CC plots VGG16MITPL365
rcParams['figure.dpi'] = 500
run_dataset_n_CC_map('gradient_x_input', 'MITPLACES365val', 'VGG16MITPL365', n=10000)
run_dataset_n_CC_map('gradient_x_sign_mu_0', 'MITPLACES365val', 'VGG16MITPL365', n=10000)
run_dataset_n_CC_map('lrpz_epsilon_0_25_std_x', 'MITPLACES365val', 'VGG16MITPL365', n=10000)
run_dataset_n_CC_map('lrpsign_epsilon_0_25_std_x_mu_0', 'MITPLACES365val', 'VGG16MITPL365', n=10000)

# Generate RV with zoomed areas, VGG16ILSVRC
rcParams['figure.dpi'] = 1000
plot_single('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o.jpg')
run_single('gradient', 'flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o.jpg', morfnum=None)
run_single('gradient_x_input', 'flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o.jpg', morfnum=None)
plot_crop_zoom('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o_img.png', cropx=2000+160, cropy=2300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=True)
plot_crop_zoom('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o_(gradient).png', cropx=2000, cropy=2300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=True)
plot_crop_zoom('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o_(gradient_x_input).png', cropx=2000, cropy=2300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=True)
plot_combine_row('flickr', 'VGG16ILSVRC', ['7867854122_b26957e9e3_o_img_crop_zoom.png', '7867854122_b26957e9e3_o_(gradient)_crop_zoom.png', '7867854122_b26957e9e3_o_(gradient_x_input)_crop_zoom.png'], '7867854122_b26957e9e3_o_row.jpg', zoomf=0.5, cleanup=True)

