from matplotlib import rcParams
from experiments.singleMoRF import run_single_MoRF

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = "\\usepackage{amssymb}\n \\usepackage{amsmath}"
rcParams['savefig.format'] = 'pdf'
brightness=1
contrast=1
filename = 'ViaduktUengsterode.jpg'

# MoRF plots VGG16ILSVRC
# run_single_MoRF('gradient', 'other', 'VGG16ILSVRC', filename)
# run_single_MoRF('gradient_x_input', 'other', 'VGG16ILSVRC', filename)
# run_single_MoRF('gradient_x_sign_mu_0', 'other', 'VGG16ILSVRC', filename)
# run_single_MoRF('lrpz_epsilon_0_5_std_x', 'other', 'VGG16ILSVRC', filename)
# run_single_MoRF('gradient_x_grad_root_diff', 'other', 'VGG16ILSVRC', filename)
# run_single_MoRF('lrpgrdtd_epsilon_0_25_std_x', 'other', 'VGG16ILSVRC', filename)
# run_single_MoRF('lrpsign_epsilon_0_5_std_x', 'other', 'VGG16ILSVRC', filename)

# for m in ['lrpgrdtd_epsilon_0_1_std_x', 'lrpgrdtd_epsilon_0_25_std_x', 'lrpgrdtd_epsilon_0_5_std_x']:
for m in ['lrpgrdtd_epsilon_0_1_std_x', 'lrpgrdtd_epsilon_0_5_std_x']:
    for lr in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200]:
        run_single_MoRF(m, 'other', 'VGG16ILSVRC', filename, lr=lr, suffix=str(lr).replace('.', '-'))

