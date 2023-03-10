import numpy as np

class MeanSquaredError:

    def __init__(self):

        self.name = "Mean Squared Error"

    def cost(self, y, y_estimate):

        error = (y_estimate - y)
        mean_squared_error = np.mean(error**2)

        return mean_squared_error

    def cost_derivative(self, y, y_estimate):

        error = (y_estimate - y)

        return error
