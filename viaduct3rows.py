from matplotlib import rcParams

from experiments.singleMoRF import run_single_MoRF

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = "\\usepackage{amssymb}\n \\usepackage{amsmath}"
rcParams['savefig.format'] = 'pdf'

# MoRF plots VGG16ILSVRC
run_single_MoRF('gradient', 'flickr', 'VGG16ILSVRC', '447888592_7e64c84851_o.jpg', morfnum=250)
# run_single_MoRF('gradient_x_input', 'flickr', 'VGG16ILSVRC', '447888592_7e64c84851_o.jpg', morfnum=250)
# run_single_MoRF('gradient_x_sign_mu_0', 'flickr', 'VGG16ILSVRC', '447888592_7e64c84851_o.jpg', morfnum=250)
# run_single_MoRF('lrpz_epsilon_0_25_std_x', 'flickr', 'VGG16ILSVRC', '447888592_7e64c84851_o.jpg', morfnum=250)
# run_single_MoRF('lrpsign_epsilon_0_25_std_x', 'flickr', 'VGG16ILSVRC', '447888592_7e64c84851_o.jpg', morfnum=250)
