import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.file import cleanup_paths, remove_filetype
from utils.preprocessing import get_image


def plot_combine_multiple(dataset_id, model_id, paths, targetfilename, ncols=3, **kwargs):
    # Derive path
    path_plots = 'plots/{}_{}'.format(dataset_id, model_id)

    # Load images
    images = [cv2.imread(p) for p in paths]

    # Generate blank image for filling purposes
    blank_image = np.ones(np.shape(images[0]), np.uint8) * 255

    # img_height, img_width = np.shape(images[0])[0:2]
    # ratio = img_height / img_width
    items_per_row = int(len(paths) / ncols)

    images_rows = []

    for i, img in enumerate(images):
        if i % items_per_row == 0:
            images_rows.append([])

        if img is not None:
            images_rows[-1].append(img)
        else:
            raise Exception('File not found: {}'.format(paths[i]))

    last_row = images_rows[-1]
    if len(last_row) < items_per_row:
        diff = items_per_row - len(last_row)

        for _ in range(diff):
            last_row.append(blank_image)

    def concat_tile(im_list_2d):
        return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

    im_tile = concat_tile(images_rows)
    cv2.imwrite('{}/{}'.format(path_plots, targetfilename), im_tile)


def plot_combine_row(dataset_id, model_id, filenames, targetfilename, zoomf=1.0, cleanup=False):
    # Derive path
    path_plots = 'plots/{}_{}'.format(dataset_id, model_id)
    path_out = '{}/{}'.format(path_plots, targetfilename)

    # Load images
    images = [cv2.imread('{}/{}'.format(path_plots, f)) for f in filenames]

    # Derive target image dimensions
    sizey, sizex = 0, 0
    for img in images:
        if img.shape[0] > sizey:
            sizey = img.shape[0]
        sizex += img.shape[1]

    # Generate blank image
    img_out = np.ones((sizey, sizex, 3), np.uint8) * 255

    posx = 0
    for img in images:
        img_out[0:img.shape[0], posx:posx+img.shape[1], :] = img
        posx = posx+img.shape[1]

    img_out = cv2.resize(img_out, None, fx=zoomf, fy=zoomf)

    # Save file
    cv2.imwrite(path_out, img_out)

    if cleanup:
        # Cleanup tmp files
        cleanup_paths(['{}/{}'.format(path_plots, f) for f in filenames])


def plot_single_pos_neg(dataset_id, model_id, filename, figdim=5, **kwargs):
    # Derive path
    path_plots = 'plots/{}_{}'.format(dataset_id, model_id)
    path_saveto = '{}/{}_img_pos_neg.png'.format(path_plots, remove_filetype(filename))

    # Create directory for plots if absent
    os.makedirs(path_plots, exist_ok=True)

    # Load and preprocess image
    img, x = get_image(filename, dataset_id, expand_dims=False)

    # Aggregate heatmap for visualization
    a = x.sum(axis=2) / 3
    mx = np.max(np.abs(a))

    # Plot image
    plt.figure(figsize=(figdim * 1.23, figdim))
    plt.imshow(a, cmap='seismic', clim=(-mx, mx))

    cb = plt.colorbar(ticks=[-mx, 0, mx])
    cb.set_label(r'$\overline{x}_{H,W}$', rotation=0, horizontalalignment="center", verticalalignment="center")
    plt.xlabel('224px')
    plt.ylabel('224px')
    plt.xticks([])
    plt.yticks([])
    plt.title('Input image (mean values)')

    plt.tight_layout()
    plt.savefig(path_saveto)
    plt.close()

    return path_saveto


def plot_single(dataset_id, model_id, filename, figdim=5, **kwargs):
    # Derive path
    path_plots = 'plots/{}_{}'.format(dataset_id, model_id)
    path_saveto = '{}/{}_img.png'.format(path_plots, remove_filetype(filename))

    # Create directory for plots if absent
    os.makedirs(path_plots, exist_ok=True)

    # Load and preprocess image
    img, x = get_image(filename, dataset_id, expand_dims=False)

    # Plot image
    plt.figure(figsize=(figdim, figdim))
    plt.imshow(img, cmap='seismic', clim=(-1, 1))

    plt.xlabel('224px')
    plt.ylabel('224px')
    plt.xticks([])
    plt.yticks([])
    plt.title('Input image')

    plt.tight_layout()
    plt.savefig(path_saveto)
    plt.close()

    return path_saveto


def plot_crop_zoom(dataset_id, model_id, filename, cropx, cropy, croph, cropw, plotx, ploty, zoomf=3.0, lw=30, cleanup=False, **kwargs):
    # Derive path
    path_plots = 'plots/{}_{}'.format(dataset_id, model_id)
    path_in = '{}/{}'.format(path_plots, filename)
    path_out = '{}/{}_crop_zoom.png'.format(path_plots, remove_filetype(filename))

    # Load image
    img_in = cv2.imread(path_in)
    if img_in is None:
        raise Exception('File not available: {}'.format(path_in))
    img_in_shape = img_in.shape

    # Color
    # lc = (72, 72, 72)
    R, G, B = (42, 180, 73)
    lc = (B, G, R)

    # Extract crop area
    crop_img = img_in[cropy:cropy + croph, cropx:cropx + cropw]

    # Create a white image
    img_out = np.concatenate([img_in, np.ones((int(img_in_shape[0]*0.35), img_in_shape[1], img_in_shape[2]), np.uint8) * 255])

    # Derive dimensions
    plotw, ploth = int(cropw * zoomf), int(croph * zoomf)

    # Draw rectangle at crop area
    cv2.rectangle(img_out, (cropx, cropy), (cropx+cropw, cropy+croph), lc, lw)

    # Draw lines between crop and plot areas
    cv2.line(img_out, (cropx, cropy), (plotx, ploty), lc, lw)
    cv2.line(img_out, (cropx+cropw, cropy), (plotx+plotw, ploty), lc, lw)
    cv2.line(img_out, (cropx, cropy+croph), (plotx, ploty+ploth), lc, lw)
    cv2.line(img_out, (cropx+cropw, cropy+croph), (plotx+plotw, ploty+ploth), lc, lw)

    # Draw zoomed crop at plot area
    img_out[ploty:ploty+ploth, plotx:plotx+plotw, :] = cv2.resize(crop_img, None, fx=zoomf, fy=zoomf)

    # Draw rectangle at plot area
    cv2.rectangle(img_out, (plotx, ploty), (plotx+plotw, ploty+ploth), lc, lw)

    # Save file
    cv2.imwrite(path_out, img_out)

    if cleanup:
        # Cleanup tmp files
        cleanup_paths([path_in])
