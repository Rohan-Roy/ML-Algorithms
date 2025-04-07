import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.00001, iterations: int = 1000):
        self.x = x
        self.y = y
        self.m = len(x)
        self.learning_rate = learning_rate
        self.iterations = iterations

    def predictions(self, slope: float, y_intercept: float) -> np.ndarray:
        """
        Calculate the predicted y values (y_hat) for given slope and y_intercept.
        
        Args:
            slope (float): The slope of the regression line.
            y_intercept (float): The y-intercept of the regression line.
        
        Returns:
            np.ndarray: Predicted y values.
        """
        return slope * self.x + y_intercept

    def calculate_error_cost(self, y_hat: np.ndarray) -> float:
        """
        Calculate the mean squared error cost.
        
        Args:
            y_hat (np.ndarray): Predicted y values.
        
        Returns:
            float: The mean squared error cost.
        """
        return (1 / (2 * self.m)) * np.sum((y_hat - self.y) ** 2)

    def gradient_descent(self) -> tuple:
        """
        Perform gradient descent to learn the slope and y_intercept.
        
        Returns:
            tuple: Final slope and y_intercept values.
        """
        costs = []
        temp_w = 0
        temp_b = 0

        for iteration in range(self.iterations):
            y_hat = self.predictions(slope=temp_w, y_intercept=temp_b)
            sum_w = np.dot(y_hat - self.y, self.x)
            sum_b = np.sum(y_hat - self.y)

            temp_w -= self.learning_rate * (1 / self.m) * sum_w
            temp_b -= self.learning_rate * (1 / self.m) * sum_b

            costs.append(self.calculate_error_cost(y_hat))

            if iteration > 0 and costs[-1] > costs[-2]:
                print(costs)
                return temp_w, temp_b

            print(iteration)

        return temp_w, temp_b


# Example usage
p = pd.read_csv('data/archive/test.csv')
x_data = p['x'].values
y_data = p['y'].values
lin_reg = LinearRegression(x_data, y_data)
y_hat = lin_reg.predictions(*lin_reg.gradient_descent())

fig = plt.figure()
plt.plot(x_data, y_data, 'r.', label='Data')
plt.plot(x_data, y_hat, 'b-', label='Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()