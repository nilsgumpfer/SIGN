import json

import numpy as np
from tensorflow.python.keras.activations import linear
from tensorflow.python.keras.models import load_model


def remove_softmax(model):
    # Remove last layer's softmax
    model.layers[-1].activation = linear

    return model


def get_models(model_id, model_dir='data/models'):
    path = '{}/{}.h5'.format(model_dir, model_id)

    model_w_softmax = load_model(path, compile=False)
    model_w_o_softmax = remove_softmax(load_model(path, compile=False))

    return model_w_softmax, model_w_o_softmax


def decode_prediction(y, model_id, model_dir='data/models'):
    path = '{}/{}labels.json'.format(model_dir, model_id)

    i = int(np.argmax(y[0]))
    p = y[0][i]

    with open(path) as fp:
        labels = json.load(fp)

    return i, labels[str(i)], round(p, 2)

def decode_prediction_top_n(y, model_id, top_n=3, model_dir='data/models'):
    path = '{}/{}labels.json'.format(model_dir, model_id)

    with open(path) as fp:
        labels_idxs = json.load(fp)

    idxs = np.argsort(-y[0], axis=0)[:top_n]
    preds = y[0][idxs]

    labels = [labels_idxs[str(idx)] for idx in idxs]

    return idxs, labels, preds
