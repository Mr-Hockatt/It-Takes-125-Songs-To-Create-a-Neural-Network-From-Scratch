from NeuronLayer import NeuronLayer
from ActivationFunctions import Sigmoid

class NeuralNetwork:

    def __init__(self, number_of_inputs, dense_layer_shape, number_of_outputs):

        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.neurons_per_layer, self.number_of_dense_layers = dense_layer_shape
        self.layers = [None] * (self.number_of_dense_layers + 1)

        self.initialize_architecture()

    def initialize_architecture(self):

        """
            This function initializes the whole network architecture.
            That is, creating each layer with it's correct dimension and
            a default activation function (Sigmoid).

            NOTES:
                - The 1st layer has the same number of input connections as the number of inputs of the system.
                - The output layer is also a layer, therefore the total number of layers is /number_of_dense_layers/ + 1.
                - Each dense layer has N neurons and N number of input connections.
        """

        self.layers[0] = NeuronLayer(self.number_of_inputs, self.neurons_per_layer, Sigmoid)

        for i in range(1, self.number_of_dense_layers):

            self.layers[i] = NeuronLayer(self.neurons_per_layer, self.neurons_per_layer, Sigmoid)

        self.layers[self.number_of_dense_layers] = NeuronLayer(self.neurons_per_layer, self.number_of_outputs, Sigmoid)

    def feed_forward(self, input_data):

        """
            Computes the /feed_forward/ function on all the network, layer by layer.

            NOTES:
                - The input for the first layer is the input of the whole system.
                - Excepting the first one, the output of the layer /i/ is the input of the layer /i + 1/.
        """

        self.layers[0].feed_forward(input_data)

        for i in range(1, self.number_of_dense_layers + 1):

            self.layers[i].feed_forward(self.layers[i - 1].output)

        return self.layers[-1].output
