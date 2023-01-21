import numpy as np

class Sigmoid:

    def __init__(self):

        self.name = "Sigmoid"

    def activation(self, x):

        return 1 / (1 + np.e**(-x))

    def activation_derivative(self, x):

        activation = self.activation(x)

        return activation * (1 - activation)

class Linear:

    def __init__(self):

        self.name = "Linear"

    def activation(self, x):

        return x

    def activation_derivative(self):

        return 1

class Tanh:

    def __init__(self):

        self.name = "Hyperbolic Tangent (aka Tanh)"

    def activation(self, x):

        e = np.e

        return (e**x - e**(-x)) / (e**x + e**(-x))

    def activation_derivative(self):

        return 1 - self.activation(x)**2

class ReLU:

    def __init__(self):

        self.name = "Rectified Linear Unit (aka ReLU)"

    def activation(self, x):

        return max(0, x)

    def activation_derivative(self):

        return np.heaviside(x, 1)

class Swish:

    def __init__(self):

        self.name = "Swish"
        self.sigmoid = Sigmoid()

    def activation(self, x):

        return x * self.sigmoid.activation(x)

    def activation_derivative(self):

        return self.sigmoid.activation(x) + x * self.sigmoid.activation_derivative(x)

class SoftPlus:

    def __init__(self):

        self.name = "SoftPlus"

    def activation(self, x):

        return np.log(1 + np.e**x)

    def activation_derivative(self):

        return 1 / (1 + np.e**(-x))

class ArcTan:

    def __init__(self):

        self.name = "ArcTan"

    def activation(self, x):

        return np.arctan(x)

    def activation_derivative(self):

        return 1 / (x**2 + 1)
