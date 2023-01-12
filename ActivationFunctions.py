import numpy as np

class Sigmoid:

    def __init__(self):

        self.name = "Sigmoid"

    def activation(self, z):

        return 1 / (1 + np.e**(-z))

    def activation_derivative(self, z):

        activation = self.activation(z)

        return activation * (1 - activation)
