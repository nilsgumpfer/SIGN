from os.path import exists

import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical


def add_shift(x, shift=1):
    collected = [x]

    fill = x[0][0]

    # right up
    y = np.roll(x, shift, axis=1)
    y[:, :shift] = fill
    y = np.roll(y, -shift, axis=0)
    y[-shift:, :] = fill
    collected.append(y)

    # left up
    y = np.roll(x, -shift, axis=1)
    y[:, -shift:] = fill
    y = np.roll(y, -shift, axis=0)
    y[-shift:, :] = fill
    collected.append(y)

    # right down
    y = np.roll(x, shift, axis=1)
    y[:, :shift] = fill
    y = np.roll(y, shift, axis=0)
    y[:shift, :] = fill
    collected.append(y)

    # left down
    y = np.roll(x, -shift, axis=1)
    y[:, -shift:] = fill
    y = np.roll(y, shift, axis=0)
    y[:shift, :] = fill
    collected.append(y)

    return np.array(collected)


def add_noise(x, prob=0.1):
    nx = np.array(x)
    mn, mx = np.min(x.ravel()), np.max(x.ravel())

    rnd = np.random.random(np.shape(x))
    nx[rnd < prob] = mn
    nx[rnd > 1 - prob] = mx

    return nx


def load_or_train_model(modelpath, net, x_train, y_train, x_test, y_test, epochs, activation='relu'):
    # Load or train model
    if exists(modelpath):
        model = load_model(modelpath)
    else:
        model = Sequential()

        if net == 'CNN':
            model.add(Conv2D(32, kernel_size=(3, 3), activation=activation, input_shape=(28, 28, 1)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, kernel_size=(3, 3), activation=activation))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dropout(0.25))
            model.add(Dense(64, activation=activation))
            model.add(Dropout(0.25))
            model.add(Dense(64, activation=activation))
            model.add(Dropout(0.25))
            model.add(Dense(64, activation=activation))
            model.add(Dropout(0.25))

        elif net == 'DENSE':
            model.add(Flatten(input_shape=(28, 28, 1)))
            model.add(Dense(512, activation=activation))
            model.add(Dropout(0.1))
            model.add(Dense(512, activation=activation))

        else:
            raise Exception('Unknown model type: {}'.format(net))

        model.add(Dense(10, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        model.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_split=0.1)

        model.save(modelpath)

    print(model.evaluate(x=x_test, y=y_test, return_dict=True))

    return model


def load_and_prepare_data(variant, inverted, translation=True, noise=True):
    (x_train, y_train), (x_test, y_test) = load_mnist()

    if translation:
        x_train = np.array([add_shift(x) for x in x_train])
        x_train = np.vstack(x_train)
        y_train = np.repeat(y_train, 5)

    if variant == '01':
        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        if inverted:
            x_train = np.ones_like(x_train) - x_train
            x_test = np.ones_like(x_test) - x_test

    elif variant == '10':
        # Scale images to the [-1, 0] range
        x_train = x_train.astype("float32") / -255.0
        x_test = x_test.astype("float32") / -255.0

        if inverted:
            x_train = -(np.ones_like(x_train) + x_train)
            x_test = -(np.ones_like(x_test) + x_test)

    elif variant == '11':
        # Scale images to the [-1, 1] range
        x_train = (x_train.astype("float32") / 127.5) - 1.0
        x_test = (x_test.astype("float32") / 127.5) - 1.0

        if inverted:
            x_train = x_train * -1
            x_test = x_test * -1

    else:
        raise Exception('Unkown variant: {}'.format(variant))

    if noise:
        x_train = np.array([add_noise(x) for x in x_train])

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def load_mnist(path='data/datasets/MNIST/mnist.npz', local=False):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)


