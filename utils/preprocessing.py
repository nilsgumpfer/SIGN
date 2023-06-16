import numpy as np
from PIL import ImageEnhance
from tensorflow.keras.preprocessing import image


def preprocess_image(x):
    # 'RGB'->'BGR'
    x = x[..., ::-1]

    # Zero-centering based on ImageNet mean RGB values
    mean = [103.939, 116.779, 123.68]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    return x


def reverse_preprocess_image(x):
    # Undo zero-centering based on ImageNet mean RGB values
    mean = [103.939, 116.779, 123.68]
    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]

    # 'BGR'->'RGB'
    x = x[..., ::-1]

    return np.array(x, dtype=int)


def get_image(filename, dataset_id, brightness=1.0, contrast=1.0, expand_dims=True):
    # Load image
    img_path = 'data/datasets/{}/{}'.format(dataset_id, filename)
    img = image.load_img(img_path, target_size=(224, 224))

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)

    # Array conversion
    x = image.img_to_array(img)

    # Adjust brightness
    x = x * brightness
    x = np.clip(x, a_min=0, a_max=255)
    img = np.array(x, dtype=int)

    if expand_dims:
        # Shape conversion
        x = np.expand_dims(x, axis=0)

    # Preprocess image model-specific
    x = preprocess_image(x)

    return img, x

