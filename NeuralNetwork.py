from ActivationFunctions import Sigmoid
from NeuronLayer import NeuronLayer
import numpy as np

class NeuralNetwork:

    def __init__(self, number_of_inputs, hidden_layer_topology, number_of_outputs, cost_function):

        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        #self.neurons_per_layer, self.number_of_hidden_layers = hidden_layer_topology
        self.hidden_layer_topology = hidden_layer_topology
        self.number_of_hidden_layers = len(self.hidden_layer_topology)
        self.layers = [None] * (self.number_of_hidden_layers + 1)
        self.cost_function = cost_function()



        self.initialize_architecture()

    def initialize_architecture(self):

        """
            This function initializes the whole network architecture.
            That is, creating each layer with it's correct dimension and
            a default activation function (Sigmoid).

            NOTES:
                - The 1st layer has the same number of input connections as the number of inputs of the system.
                - Each hidden layer has N neurons and N number of input connections.
                - The output layer is also a layer, therefore the total number of layers is /number_of_hidden_layers/ + 1.
        """

        full_topology = [self.number_of_inputs] + self.hidden_layer_topology + [self.number_of_outputs]

        for i in range(self.number_of_hidden_layers + 1):

            number_of_inputs = full_topology[i]
            number_of_outputs = full_topology[i + 1]

            self.layers[i] = NeuronLayer(number_of_inputs, number_of_outputs, Sigmoid)

        """
        self.layers[0] = NeuronLayer(self.number_of_inputs, self.neurons_per_layer, Sigmoid)

        for i in range(1, self.number_of_hidden_layers):

            self.layers[i] = NeuronLayer(self.neurons_per_layer, self.neurons_per_layer, Sigmoid)

        self.layers[self.number_of_hidden_layers] = NeuronLayer(self.neurons_per_layer, self.number_of_outputs, Sigmoid)
        """

    def feed_forward(self, input_data):

        """
            Computes the /feed_forward/ function on all the network, layer by layer.

            NOTES:
                - The input for the first layer is the input of the whole system.
                - Excepting the first one, the output of the layer /i/ is the input of the layer /i + 1/.
        """

        self.layers[0].feed_forward(input_data)

        for i in range(1, self.number_of_hidden_layers + 1):

            self.layers[i].feed_forward(self.layers[i - 1].activation)

        return self.layers[-1].activation

    def update_weights(self, l, dC_dWl, alpha):

        """
            Implements the update formula for Backpropagation algorithm.

            W_l = W_l - alpha * dC_dWl

            Where:

                l: Layer index
                alpha: Learning rate
                W_l: Weights in layer l
                dC_dWl: Partial derivative of the cost function with respect to W_l
        """

        self.layers[l].weights = self.layers[l].weights - alpha * dC_dWl

    def update_biases(self, l, dC_dBl, alpha):

        """
            Implements the update formula for Backpropagation algorithm.

            B_l = B_l - alpha * dC_dBl

            Where:

                l: Layer index
                alpha: Learning rate
                B_l: Biases in layer l
                dC_dBl: Partial derivative of the cost function with respect to B_l
        """

        self.layers[l].biases = self.layers[l].biases - alpha * np.mean(dC_dBl, axis=0).reshape(self.layers[l].biases.shape)

    def back_propagate(self, input_data, output_data, learning_rate):

        """
            Backpropagates the error, layer by layer in a Gradient Descent fashion.
        """

        dC_dAl = self.cost_function.cost_derivative(output_data, self.layers[-1].activation)
        dAl_dZl = self.layers[-1].activation_function.activation_derivative(self.layers[-1].output)
        dZl_dWl = self.layers[-2].activation

        delta_l = dC_dAl * dAl_dZl

        self.update_biases(-1, delta_l, learning_rate)

        dC_dWl = dZl_dWl.T @ delta_l

        for l in reversed(range(1, self.number_of_hidden_layers)):

            dZlplus1_dAl = self.layers[l + 1].weights
            dAl_dZl = self.layers[l].activation_function.activation_derivative(self.layers[l].output)
            dZl_dWl = self.layers[l - 1].activation

            delta_l = delta_l @ dZlplus1_dAl.T * dAl_dZl

            self.update_biases(l, delta_l, learning_rate)

            self.update_weights(l + 1, dC_dWl, learning_rate)

            dC_dWl = dZl_dWl.T @ delta_l

        l = 0

        dZlplus1_dAl = self.layers[l + 1].weights
        dAl_dZl = self.layers[l].activation_function.activation_derivative(self.layers[l].output)
        dZl_dWl = input_data

        delta_l = delta_l @ dZlplus1_dAl.T * dAl_dZl
        self.update_biases(l, delta_l, learning_rate)

        self.update_weights(l + 1, dC_dWl, learning_rate)

        dC_dWl = dZl_dWl.T @ delta_l

        self.update_weights(l, dC_dWl, learning_rate)
