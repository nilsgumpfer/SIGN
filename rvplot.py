from matplotlib import rcParams
rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = "\\usepackage{amssymb}\n \\usepackage{amsmath}"
rcParams['savefig.format'] = 'pdf'

from experiments.single import run_single
from utils.image import plot_single, plot_crop_zoom, plot_combine_row, plot_single_pos_neg

# Generate RV with zoomed areas, VGG16ILSVRC
rcParams['figure.dpi'] = 1000
plot_single('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o.jpg')
run_single('gradient', 'flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o.jpg', morfnum=None)
run_single('gradient_x_input', 'flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o.jpg', morfnum=None)
run_single('gradient_x_sign', 'flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o.jpg', morfnum=None)
plot_single_pos_neg('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o.jpg')
plot_crop_zoom('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o_img.png', cropx=2000+160, cropy=2300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=True)
plot_crop_zoom('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o_(gradient).png', cropx=2000, cropy=2300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=True)
plot_crop_zoom('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o_(gradient_x_input).png', cropx=2000, cropy=2300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=True)
plot_crop_zoom('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o_(gradient_x_sign).png', cropx=2000, cropy=2300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=True)
plot_crop_zoom('flickr', 'VGG16ILSVRC', '7867854122_b26957e9e3_o_img_pos_neg.png', cropx=2000, cropy=2300, croph=600, cropw=600, ploty=4800, plotx=20, cleanup=True)
plot_combine_row('flickr', 'VGG16ILSVRC', ['7867854122_b26957e9e3_o_img_crop_zoom.png', '7867854122_b26957e9e3_o_img_pos_neg_crop_zoom.png', '7867854122_b26957e9e3_o_(gradient)_crop_zoom.png', '7867854122_b26957e9e3_o_(gradient_x_input)_crop_zoom.png', '7867854122_b26957e9e3_o_(gradient_x_sign)_crop_zoom.png'], '7867854122_b26957e9e3_o_row.jpg', zoomf=0.5, cleanup=False)

