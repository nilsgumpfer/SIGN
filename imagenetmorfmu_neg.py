from utils.env import set_visible_gpu

set_visible_gpu(1)

from experiments.bulk import run_bulk
from matplotlib import rcParams

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = "\\usepackage{amssymb}\n \\usepackage{amsmath}"
rcParams['savefig.format'] = 'pdf'
rcParams['figure.dpi'] = 1000

methods = ['gradient_x_sign_mu_neg_32',
           'gradient_x_sign_mu_neg_64',
           'gradient_x_sign_mu_neg_96',
           'gradient_x_sign_mu_neg_128']

# Run experiments
run_bulk(dataset_id='ILSVRC2012val', model_id='VGG16ILSVRC', methods=methods, calc_pcc=False, calc_scc=False)
