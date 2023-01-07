import numpy as np

class MeanSquaredError:

    def __init__(self):

        self.name = "Mean Squared Error"

    def cost(self, y, y_estimate):

        error = (y_estimate - y)
        mean_squared_error = (1/2) * np.mean(error**2, axis=0)

        return mean_squared_error

    def cost_derivative(self, y, y_estimate):

        error = (y_estimate - y)

        return np.mean(error, axis=0)
