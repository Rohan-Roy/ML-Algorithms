import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self , x:np.ndarray ,y:np.ndarray):
        self.x = x
        self.m = len(x)
        self.y = y

    # Calculate y hat.
    def predictions(self ,slope:int , y_intercept:int) -> np.ndarray: 
        predictions = []

        for x in self.x:
            predictions.append(slope * x + y_intercept)

        return predictions

    def calculate_error_cost(self , y_hat:np.ndarray) -> int:
        error_values = []
        for i in range(self.m):
            error_values.append((y_hat[i] - self.y[i] )** 2)

        error = (1/(2*self.m)) * sum(error_values)
    
        return error
    
    def gradient_descent(self):
        costs = []

        # initialization values        
        temp_w = 0
        temp_b = 0
        iteration = 0
        
        # Learning rate
        a = 0.00001 

        while iteration < 1000:
            y_hat = self.predictions(slope=temp_w , y_intercept= temp_b)
            
            sum_w = 0
            sum_b = 0

            for i in range(len(self.x)):
                sum_w += (y_hat[i] - self.y[i] ) * self.x[i]
                sum_b += (y_hat[i] - self.y[i] )

            w = temp_w - a * ((1/self.m) *sum_w)
            b = temp_b - a * ((1/self.m) *sum_b)

            costs.append(self.calculate_error_cost(y_hat))

            try:
                if costs[-1] > costs[-2]: # If global minimum reached
                    print(costs)
                    return [temp_w,temp_b]
            except IndexError:
                pass

            temp_w = w
            temp_b = b
            iteration += 1
            print(iteration)

        return [temp_w,temp_b]

p = pd.read_csv('data/archive/test.csv')

x_data = p['x']
y_data = p['y']
lin_reg = LinearRegression(x_data, y_data)
y_hat = lin_reg.predictions(*lin_reg.gradient_descent())

fig = plt.figure()
plt.plot(x_data, y_data, 'r.', label='Data')
plt.plot(x_data, y_hat, 'b-', label='Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()