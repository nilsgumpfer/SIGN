import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def compute_rgb_distributions(image_dir):
    """
    Compute the RGB distributions for all images in a directory.

    Args:
        image_dir (str): Path to the directory containing the images.

    Returns:
        np.ndarray: Arrays containing the counts of pixel intensities for R, G, and B channels.
    """
    # Initialize histograms for R, G, and B channels
    bins = 256  # 0-255 intensity values
    r_hist = np.zeros(bins, dtype=np.int64)
    g_hist = np.zeros(bins, dtype=np.int64)
    b_hist = np.zeros(bins, dtype=np.int64)

    # Iterate over all images in the directory
    for i, filename in enumerate(os.listdir(image_dir)):
        print(i)
        file_path = os.path.join(image_dir, filename)

        # Ensure the file is an image
        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')  # Convert to RGB if not already
                img_array = np.asarray(img)

                # Flatten the image array and update histograms
                r_hist += np.bincount(img_array[:, :, 0].flatten(), minlength=bins)
                g_hist += np.bincount(img_array[:, :, 1].flatten(), minlength=bins)
                b_hist += np.bincount(img_array[:, :, 2].flatten(), minlength=bins)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    return r_hist, g_hist, b_hist


def plot_rgb_distributions(r_hist, g_hist, b_hist):
    """
    Plot the RGB distributions.

    Args:
        r_hist (np.ndarray): Red channel histogram.
        g_hist (np.ndarray): Green channel histogram.
        b_hist (np.ndarray): Blue channel histogram.
    """
    bins = np.arange(256)  # Pixel intensity values

    plt.figure(figsize=(10, 6))
    plt.plot(bins, r_hist, color='red', label='Red')
    plt.plot(bins, g_hist, color='green', label='Green')
    plt.plot(bins, b_hist, color='blue', label='Blue')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('RGB Channel Distributions')
    plt.legend()
    plt.grid()
    plt.savefig('./plots/imagenet.pdf')


if __name__ == '__main__':
    # Path to the directory containing the images
    image_directory = './data/datasets/ILSVRC2012val'

    # Compute RGB distributions
    r_histogram, g_histogram, b_histogram = compute_rgb_distributions(image_directory)

    # Plot RGB distributions
    plot_rgb_distributions(r_histogram, g_histogram, b_histogram)
