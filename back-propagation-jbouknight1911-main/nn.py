"""
The main code for the back propagation assignment. See README.md for details.
"""
import math
from typing import List

import numpy as np
import scipy


class SimpleNetwork:
    """A simple feedforward network where all units have sigmoid activation.
    """

    @classmethod
    def random(cls, *layer_units: int):
        """Creates a feedforward neural network with the given number of units
        for each layer.

        :param layer_units: Number of units for each layer
        :return: the neural network
        """

        def uniform(n_in, n_out):
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            return np.random.uniform(-epsilon, +epsilon, size=(n_in, n_out))

        pairs = zip(layer_units, layer_units[1:])
        return cls(*[uniform(i, o) for i, o in pairs])

    def __init__(self, *layer_weights: np.ndarray):
        """Creates a neural network from a list of weight matrices.
        The weights correspond to transformations from one layer to the next, so
        the number of layers is equal to one more than the number of weight
        matrices.

        :param layer_weights: A list of weight matrices
        """

        self.layer_weights = layer_weights

    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix.

        Each unit's output should be calculated by taking a weighted sum of its
        inputs (using the appropriate weight matrix) and passing the result of
        that sum through a logistic sigmoid activation function.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each in the range (0, 1) - for the corresponding row in the
        input matrix.
        """

        sigmoid_activation = input_matrix

        for weight in self.layer_weights:
            weighted_sum = sigmoid_activation.dot(weight)
            sigmoid_activation = scipy.special.expit(weighted_sum)

        return sigmoid_activation

    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).

        Outputs will be converted to 0 if they are less than 0.5, and converted
        to 1 otherwise.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each either 0 or 1 - for the corresponding row in the input
        matrix.
        """

        matrix = self.predict(input_matrix)
        binary_matrix = np.where(matrix < 0.5, 0, 1)

        return binary_matrix

    def gradients(self,
                  input_matrix: np.ndarray,
                  output_matrix: np.ndarray) -> List[np.ndarray]:
        """Performs back-propagation to calculate the gradients for each of
        the weight matrices.

        This method first performs a pass of forward propagation through the
        network, then applies the following procedure to calculate the
        gradients. In the following description, × is matrix multiplication,
        ⊙ is element-wise product, and ⊤ is matrix transpose.

        First, calculate the error, error_L, between last layer's activations,
        h_L, and the output matrix, y:

        error_L = h_L - y

        Then, for each layer l in the network, starting with the layer before
        the output layer and working back to the first layer (the input matrix),
        calculate the gradient for the corresponding weight matrix as follows.
        First, calculate g_l as the element-wise product of the error for the
        next layer, error_{l+1}, and the sigmoid gradient of the next layer's
        weighted sum (before the activation function), a_{l+1}.

        g_l = (error_{l+1} ⊙ sigmoid'(a_{l+1}))⊤

        Then calculate the gradient matrix for layer l as the matrix
        multiplication of g_l and the layer's activations, h_l, divided by the
        number of input examples, N:

        grad_l = (g_l × h_l)⊤ / N

        Finally, calculate the error that should be backpropagated from layer l
        as the matrix multiplication of the weight matrix for layer l and g_l:

        error_l = (weights_l × g_l)⊤

        Once this procedure is complete for all layers, the grad_l matrices
        are the gradients that should be returned.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :return: two matrices of gradients, one for the input-to-hidden weights
        and one for the hidden-to-output weights
        """

        error_l = self.predict(input_matrix) - output_matrix

        sums = self.layers_and_sums(input_matrix)[0]
        layers = self.layers_and_sums(input_matrix)[1]

        layer_gradients = []

        layer = len(layers)-2

        while layer >= 0:
            sig_act = scipy.special.expit(sums[layer])
            sig_act_prime = sig_act * (1-sig_act)

            g_l = np.transpose(np.multiply(error_l, sig_act_prime))
            grad_l = np.transpose(np.matmul(g_l, layers[layer])) / len(input_matrix)
            error_l = np.transpose(np.matmul(self.layer_weights[layer], g_l))

            layer_gradients.insert(0, grad_l)

            layer -= 1

        return layer_gradients

    def layers_and_sums(self, input_matrix: np.array) -> np.array:
        """
        Similar to predict method, but is keeping track of all layers and all weighted
        sums. This is a helper method which is used in the gradients() method. Returns a tuple
        of the layers and weighted sums.
        :param input_matrix:
        :return: tuple of activation layers and weighted sums
        """
        activation_layers = [input_matrix]
        weighted_sums = []

        activation = input_matrix

        for weight in self.layer_weights:
            weighted_sum = activation.dot(weight)
            weighted_sums.append(weighted_sum)

            activation = scipy.special.expit(weighted_sum)
            activation_layers.append(activation)

        return weighted_sums, activation_layers

    def train(self,
              input_matrix: np.ndarray,
              output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output
        matrix.

        Training should repeatedly (`iterations` times) calculate the gradients,
        and update the model by subtracting the learning rate times the
        gradients from the model weight matrices.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :param iterations: The number of gradient descent steps to take.
        :param learning_rate: The size of gradient descent steps to take, a
        number that the gradients should be multiplied by before updating the
        model weights.
        """

        model_weights = list(self.layer_weights)

        for _ in range(0, iterations):
            grads = self.gradients(input_matrix, output_matrix)

            for j in range(0, len(self.layer_weights)):
                model_weights[j] = model_weights[j] - (grads[j] * learning_rate)

                self.layer_weights = tuple(model_weights)
