import pandas as pd
import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy.linalg as linalg

data_train = pd.read_csv('regression_train.csv')

data_train

x_train = data_train['x']
y_train = data_train['y']

X = np.column_stack((np.ones(x_train.shape), x_train))
X

A = np.array([[1, 0.5], [0.5, 1]])
a = np.array([[1], [0]])

# specify data points for x0 and x1 (from - 5 to 5, using 51 uniformly distributed points)
x0Array = np.linspace(-5, 5, 51)
x1Array = np.linspace(-5, 5, 51)

Earray = np.zeros((51,51))

for i in range(0,50):
    for j in range(0,50):
        
        x = np.array([[x0Array[i]], [x1Array[j]]])
        tmp = a - 5 * x
        
        Earray[i,j] = tmp.transpose().dot(A).dot(tmp)

XX = X.transpose().dot(X)

#w = np.linalg.solve(XX, X.transpose().dot(y_train))
w = np.linalg.inv(XX).dot(X.transpose().dot(y_train))

w

Xtest = np.column_stack((np.ones(x_train.shape), x_train))
ytest_predicted = Xtest.dot(w)

def getPolynomialDataMatrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1,degree + 1):
        X = np.column_stack((X, x ** i))
    return X
    
print(getPolynomialDataMatrix(x_train, 4))