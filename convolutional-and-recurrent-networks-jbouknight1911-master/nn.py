"""
The main code for the recurrent and convolutional networks assignment.
See README.md for details.
"""
from typing import Tuple, List, Dict

import tensorflow


def create_toy_rnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a recurrent neural network for a toy problem.

    The network will take as input a sequence of number pairs, (x_{t}, y_{t}),
    where t is the time step. It must learn to produce x_{t-3} - y{t} as the
    output of time step t.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """

    toy = tensorflow.keras.Sequential()
    toy.add(tensorflow.keras.layers.SimpleRNN(units=300, activation='linear',
                                              input_shape=input_shape,
                                              return_sequences=True))

    for _ in range(6):
        toy.add(tensorflow.keras.layers.Dense(600, activation='linear'))

    toy.add(tensorflow.keras.layers.Dense(n_outputs, activation='linear'))
    toy.compile(optimizer='adagrad', loss='mse')

    dictionary = {'batch_size': 200}

    result = toy, dictionary

    return result


def create_mnist_cnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a convolutional neural network for digit classification.

    The network will take as input a 28x28 grayscale image, and produce as
    output one of the digits 0 through 9. The network will be trained and tested
    on a fraction of the MNIST data: http://yann.lecun.com/exdb/mnist/

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """

    cnn = tensorflow.keras.Sequential()
    cnn.add(tensorflow.keras.layers.Conv2D(filters=3, kernel_size=3,
                                           activation='tanh',
                                           input_shape=input_shape))

    for _ in range(2):
        cnn.add(tensorflow.keras.layers.Conv2D(filters=30, kernel_size=3, activation='tanh'))
        cnn.add(tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2)))

    cnn.add(tensorflow.keras.layers.Flatten())
    cnn.add(tensorflow.keras.layers.Dense(n_outputs, activation='softmax'))
    cnn.compile(optimizer='adam', loss='categorical_crossentropy')

    dictionary = {'batch_size': 2}

    result = (cnn, dictionary)

    return result


def create_youtube_comment_rnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a recurrent neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """

    yt_rnn = tensorflow.keras.Sequential()
    yt_rnn.add(tensorflow.keras.layers.Embedding(input_dim=len(vocabulary), output_dim=10))

    yt_rnn.add(
        tensorflow.keras.layers.Bidirectional(
            tensorflow.keras.layers.LSTM(units=20,
                                         activation='tanh',
                                         return_sequences=False)))

    for _ in range(3):
        yt_rnn.add(tensorflow.keras.layers.Dense(30, activation='tanh'))

    yt_rnn.add(tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid'))
    yt_rnn.compile(optimizer='adam', loss='binary_crossentropy')

    dictionary = {'batch_size': 2}

    result = (yt_rnn, dictionary)

    return result


def create_youtube_comment_cnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a convolutional neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """

    yt_cnn = tensorflow.keras.Sequential()

    yt_cnn.add(tensorflow.keras.layers.Embedding(input_dim=len(vocabulary), output_dim=400))

    yt_cnn.add(tensorflow.keras.layers.Conv1D(filters=400, kernel_size=2,
                                              strides=2,
                                              input_shape=(2, 2)))

    yt_cnn.add(tensorflow.keras.layers.GlobalMaxPool1D())

    yt_cnn.add(tensorflow.keras.layers.Flatten())
    yt_cnn.add(tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid'))

    yt_cnn.compile(optimizer='adam', loss='binary_crossentropy')

    dictionary = {'batch_size': 2}

    result = (yt_cnn, dictionary)

    return result
