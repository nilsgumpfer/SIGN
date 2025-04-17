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
# run_single_MoRF('gradient', 'other', 'VGG16ILSVRC', filename, morfnum=250, brightness=brightness, contrast=contrast)
# run_single_MoRF('gradient_x_input', 'other', 'VGG16ILSVRC', filename, morfnum=250, brightness=brightness, contrast=contrast)
# run_single_MoRF('gradient_x_sign_mu_0', 'other', 'VGG16ILSVRC', filename, morfnum=250, brightness=brightness, contrast=contrast)
run_single_MoRF('lrpz_epsilon_0_5_std_x', 'other', 'VGG16ILSVRC', filename, morfnum=250, brightness=brightness, contrast=contrast)
# run_single_MoRF('lrpsign_epsilon_0_5_std_x', 'other', 'VGG16ILSVRC', filename, morfnum=250, brightness=brightness, contrast=contrast)
