from matplotlib import rcParams
rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = "\\usepackage{amssymb}\n \\usepackage{amsmath}"
rcParams['savefig.format'] = 'pdf'
rcParams['figure.dpi'] = 1000

from experiments.single import run_single
from utils.image import plot_single, plot_crop_zoom, plot_combine_row, plot_single_pos_neg


methods = ['gradient',
           'gradient_x_input',
           'gradient_x_sign',
           'lrpz_epsilon_0_1_std_x',
           'lrpsign_epsilon_0_1_std_x',
           'gradient_x_grad_root_diff']

filenames = []
plot_single('other', 'VGG16ILSVRC', 'ViaduktUengsterode.jpg')
filenames += plot_crop_zoom('other', 'VGG16ILSVRC', 'ViaduktUengsterode_img.png', cropx=2000+160-500, cropy=1300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=False)

plot_single_pos_neg('other', 'VGG16ILSVRC', 'ViaduktUengsterode.jpg')
filenames += plot_crop_zoom('other', 'VGG16ILSVRC', 'ViaduktUengsterode_img_pos_neg.png', cropx=2000-500, cropy=1300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=False)

for m in methods:
    print(m)
    run_single(m, 'other', 'VGG16ILSVRC', 'ViaduktUengsterode.jpg')
    filenames += plot_crop_zoom('other', 'VGG16ILSVRC', 'ViaduktUengsterode_({}).png'.format(m), cropx=2000-500, cropy=1300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=False)

# plot_combine_row('other', 'VGG16ILSVRC', ['ViaduktUengsterode_img_crop_zoom.png', 'ViaduktUengsterode_img_pos_neg_crop_zoom.png', 'ViaduktUengsterode_(gradient_x_input)_crop_zoom.png'], 'ViaduktUengsterode_row.jpg', zoomf=0.5, cleanup=False)
plot_combine_row('other', 'VGG16ILSVRC', filenames, 'ViaduktUengsterode_row_2.jpg', zoomf=0.5, cleanup=False)

