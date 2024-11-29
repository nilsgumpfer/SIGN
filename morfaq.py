from matplotlib import rcParams

from experiments.singleMoRF import run_single_MoRF

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = "\\usepackage{amssymb}\n \\usepackage{amsmath}"
rcParams['savefig.format'] = 'pdf'

# Generate RV with zoomed areas, VGG16ILSVRC
rcParams['figure.dpi'] = 1000
# run_single_MoRF('gradient', 'flickr', 'VGG16ILSVRC', '447888592_7e64c84851_o.jpg', morfnum=250)
# run_single_MoRF('smoothgrad', 'flickr', 'VGG16ILSVRC', '447888592_7e64c84851_o.jpg', morfnum=250)
# run_single_MoRF('integrated_gradients', 'flickr', 'VGG16ILSVRC', '447888592_7e64c84851_o.jpg', morfnum=250)

methods = ['gradient',
           'gradient_x_sign_mu_neg_128',
           'gradient_x_sign_mu_neg_96',
           'gradient_x_sign_mu_neg_64',
           'gradient_x_sign_mu_neg_32',
           'gradient_x_sign_mu_0',
           'gradient_x_sign_mu_32',
           'gradient_x_sign_mu_64',
           'gradient_x_sign_mu_96',
           'gradient_x_sign_mu_128']

for m in methods:
    run_single_MoRF(m, 'flickr', 'VGG16ILSVRC', '447888592_7e64c84851_o.jpg', morfnum=250)