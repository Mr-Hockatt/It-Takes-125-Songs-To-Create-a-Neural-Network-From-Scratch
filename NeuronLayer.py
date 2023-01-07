import numpy as np

class NeuronLayer:

    """
        Class definition for a Neural Network Layer.
        It comprises a function to compute the output z,
        a function to pass this output through it's layer activation function
        and another one to combine both of them being it the feed forward behaviour of the layer.
    """

    def __init__(self, number_of_inputs, number_of_neurons, activation_function):

        self.weights = np.random.rand(number_of_inputs, number_of_neurons)
        self.biases = np.zeros((1, number_of_neurons))
        self.activation_function = activation_function()

    def compute_output(self, input_data):

        """
            Performs the output of the layer as follows:

            z = x * w + b

            Where

            z: Neuron's output (without activation function)
            x: Input data
            w: Neurons weights
            b: Neurons biases
        """

        self.output = input_data @ self.weights + self.biases

        return self.output

    def activate(self, z):

        """
            Performs the activation function on the Neurons output z.
        """

        self.activation = self.activation_function.activation(z)

        return self.activation

    def feed_forward(self, input_data):

        """
            Combines the /compute_output/ and /activate/ functions above. Computes the Neurons output
            and then passes it through the activation function.
        """

        z = self.compute_output(input_data)
        y = self.activate(z)

        return y
