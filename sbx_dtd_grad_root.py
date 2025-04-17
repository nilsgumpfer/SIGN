from matplotlib import rcParams
from experiments.singleMoRF import run_single_MoRF

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = "\\usepackage{amssymb}\n \\usepackage{amsmath}"
rcParams['savefig.format'] = 'pdf'

# dataset_id = 'flickr'
dataset_id = 'other'

filename = 'ViaduktUengsterode.jpg'
# filename = '234775821_ae4ebf6f2a_o.jpg'
# filename = '447888592_7e64c84851_o.jpg'
# filename = '234775821_ae4ebf6f2a_o.jpg'
# filename = '3842551283_c289a49691_o.jpg'
# filename = '605505232_e93d1a976f_o.jpg'
# filename = '15519846401_582d9bcbac_o.jpg'
# filename = '9527376792_66fd86567e_o.jpg'
# filename = 'baseballplayer.png'

# method = 'gradient_x_grad_root_diff'
# method = 'lrpgrdtd_epsilon_0_5_std_x'
method = 'lrpgrdtd_epsilon_0_25_std_x'
# method = 'lrpsign_epsilon_0_5_std_x'

lr = 10
# lr = 100

run_single_MoRF(method, dataset_id, 'VGG16ILSVRC', filename, lr=lr, suffix=str(lr).replace('.', '-'))