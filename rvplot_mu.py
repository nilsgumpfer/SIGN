from matplotlib import rcParams
rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = "\\usepackage{amssymb}\n \\usepackage{amsmath}"
rcParams['savefig.format'] = 'pdf'
rcParams['figure.dpi'] = 1000

from experiments.single import run_single
from utils.image import plot_single, plot_crop_zoom, plot_combine_row, plot_single_pos_neg


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

filenames = []
plot_single('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o.jpg')
filenames += plot_crop_zoom('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o_img.png', cropx=2000+160, cropy=2300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=True)

plot_single_pos_neg('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o.jpg')
filenames += plot_crop_zoom('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o_img_pos_neg.png', cropx=2000, cropy=2300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=True)

for m in methods:
    print(m)
    run_single(m, 'flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o.jpg')
    filenames += plot_crop_zoom('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o_({}).png'.format(m), cropx=2000, cropy=2300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=True)

plot_combine_row('flickr', 'VGG16ILSVRC', filenames, '7867854122_b26957e9e3_o_row_mu_exp.jpg', zoomf=0.5, cleanup=False)

