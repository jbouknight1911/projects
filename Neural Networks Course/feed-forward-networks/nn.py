"""
The main code for the feedforward networks assignment.
See README.md for details.
"""
from typing import Tuple, Dict

import tensorflow

def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters and the same activation functions.

    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """

    deep = tensorflow.keras.Sequential()
    deep.add(tensorflow.keras.layers.Dense(300, activation='linear', input_dim=n_inputs))

    for _ in range(4):
        deep.add(tensorflow.keras.layers.Dense(600, activation='linear'))

    deep.add(tensorflow.keras.layers.Dense(n_outputs, activation='linear'))
    deep.compile(optimizer='adam', loss='mse')

    wide = tensorflow.keras.Sequential()
    wide.add(tensorflow.keras.layers.Dense(805, activation='linear', input_dim=n_inputs))
    wide.add(tensorflow.keras.layers.Dense(1450, activation='linear'))
    wide.add(tensorflow.keras.layers.Dense(n_outputs, activation='linear'))
    wide.compile(optimizer='adam', loss='mse')

    networks = (deep, wide)

    return networks


def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network where all hidden layers have ReLU activations,
    and one where all hidden layers have tanh activations. The networks should
    be identical other than the difference in activation functions.

    The neural networks will be asked to predict the 0 or more tags associated
    with a del.icio.us bookmark. They will be trained and tested on the
    del.icio.us dataset from:
    https://github.com/dhruvramani/Multilabel-Classification-Datasets
    which is a slightly simplified version of:
    https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (ReLU neural network, tanh neural network)
    """

    tanh = tensorflow.keras.Sequential()
    tanh.add(tensorflow.keras.layers.Dense(1000, activation='tanh', input_dim=n_inputs))
    tanh.add(tensorflow.keras.layers.Dense(700, activation='tanh'))
    tanh.add(tensorflow.keras.layers.Dense(600, activation='tanh'))
    tanh.add(tensorflow.keras.layers.Dense(100, activation='tanh'))
    tanh.add(tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid'))
    tanh.compile(optimizer='adam', loss='binary_crossentropy')
    tanh.summary()

    relu = tensorflow.keras.Sequential()
    relu.add(tensorflow.keras.layers.Dense(1000, activation='relu', input_dim=n_inputs))
    relu.add(tensorflow.keras.layers.Dense(700, activation='relu'))
    relu.add(tensorflow.keras.layers.Dense(600, activation='relu'))
    relu.add(tensorflow.keras.layers.Dense(100, activation='relu'))
    relu.add(tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid'))
    relu.compile(optimizer='adam', loss='binary_crossentropy')
    relu.summary()

    networks = (relu, tanh)

    return networks



def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """

    dropout = tensorflow.keras.Sequential()
    dropout.add(tensorflow.keras.layers.Dense(600, activation='linear', input_dim=n_inputs))
    for _ in range(4):
        dropout.add(tensorflow.keras.layers.Dense(600, activation='linear'))
        dropout.add(tensorflow.keras.layers.Dropout(0.30))

    dropout.add(tensorflow.keras.layers.Dense(n_outputs, activation='softmax'))
    dropout.compile(optimizer='adam', loss='categorical_crossentropy')

    nondropout = tensorflow.keras.Sequential()
    nondropout.add(tensorflow.keras.layers.Dense(600, activation='linear', input_dim=n_inputs))
    for _ in range(4):
        nondropout.add(tensorflow.keras.layers.Dense(600, activation='linear'))

    nondropout.add(tensorflow.keras.layers.Dense(n_outputs, activation='softmax'))
    nondropout.compile(optimizer='adam', loss='categorical_crossentropy')

    networks = (dropout, nondropout)

    return networks



def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                Dict,
                                                tensorflow.keras.models.Model,
                                                Dict]:
    """Creates one neural network that uses early stopping during training, and
    one that does not. The networks should be identical other than the presence
    or absence of early stopping.

    The neural networks will be asked to predict whether a person makes more
    than $50K per year. They will be trained and tested on the "adult" dataset
    from:
    https://archive.ics.uci.edu/ml/datasets/adult

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
        no-early-stopping neural network,
        no-early-stopping parameters that should be passed to Model.fit
    )
    """

    early = tensorflow.keras.Sequential()
    early.add(tensorflow.keras.layers.Dense(1000, activation='relu', input_dim=n_inputs))
    early.add(tensorflow.keras.layers.Dense(700, activation='relu'))
    early.add(tensorflow.keras.layers.Dense(600, activation='relu'))
    early.add(tensorflow.keras.layers.Dense(100, activation='relu'))
    early.add(tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid'))
    early.compile(optimizer='adam', loss='binary_crossentropy')

    keys = {'callbacks': tensorflow.keras.callbacks.EarlyStopping()}

    early.summary()

    nonearly = tensorflow.keras.Sequential()
    nonearly.add(tensorflow.keras.layers.Dense(1000, activation='relu', input_dim=n_inputs))
    nonearly.add(tensorflow.keras.layers.Dense(700, activation='relu'))
    nonearly.add(tensorflow.keras.layers.Dense(600, activation='relu'))
    nonearly.add(tensorflow.keras.layers.Dense(100, activation='relu'))
    nonearly.add(tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid'))
    nonearly.compile(optimizer='adam', loss='binary_crossentropy')

    dictionary = {}

    networks_and_params = (early, keys, nonearly, dictionary)

    return networks_and_params
