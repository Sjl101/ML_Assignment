# Change all this as its toms
# Import the libraries used:
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

data_frame = pd.read_csv('pol_regression.csv')
x_train = data_frame['x']
y_train = data_frame['y']

def feature_expansion(features, degree):
    X = np.ones(features.shape)

    for i in range(1, degree + 1):
        X = np.column_stack((X, features ** i))

    return X

def pol_regression(x, y, degree):
    X = feature_expansion(x, degree=degree)

    # Least Square:
    first_half = X.transpose().dot(X)
    weights = np.linalg.solve(first_half, X.transpose().dot(y))
    
    return weights

def calculate_y(x, weights):
    total = 0
    for i, weight in enumerate(weights):        
        val = weight * (x ** i)
        total += val
    return total

def plot_predictions(weights, features):
    y =[]
    for x in features:
        y.append(calculate_y(x, weights))

    return y
    


#wd0 = pol_regression(x_train, y_train, degree=0)
#y0 = plot_predictions(weights=wd0, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
#plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y0, color='green')

# Calculates the weights:
wd1 = pol_regression(x_train, y_train, degree=1)
# Calculates predicted:
y1 = plot_predictions(weights=wd1, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
# Plots the data on the graph
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y1, color='red')

wd2 = pol_regression(x_train, y_train, degree=2)
y2 = plot_predictions(weights=wd2, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y2, color='cyan')

wd3 = pol_regression(x_train, y_train, degree=3)
y3 = plot_predictions(weights=wd3, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y3, color='yellow')

wd6 = pol_regression(x_train, y_train, degree=6)
y6 = plot_predictions(weights=wd6, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y6, color='violet')

wd10 = pol_regression(x_train, y_train, degree=10)
y10 = plot_predictions(weights=wd10, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y10, color='lime')

plt.ylim([-150, 50])
plt.xlim([-4, 4])
plt.scatter(x_train, y_train)
plt.legend(('Degree - 1', "Degree - 2", "Degree - 3", "Degree - 6", "Degree - 10", 'Training Points'), loc = 'lower right')
plt.show()
