import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy.linalg as linalg

data_frame = pd.read_csv('pol_regression.csv')
x_train = data_frame['x']
y_train = data_frame['y']
train_df, test_df = train_test_split(data_frame, test_size=0.3)
x_train = train_df['x']
y_train = train_df['y']
x_test = test_df['x']
y_test = test_df['y']
x_train = x_train.sort_values(ascending=True)
y_train = y_train.sort_values(ascending=True)
x_test = x_train.sort_values(ascending=True)
y_test = y_train.sort_values(ascending=True)

import matplotlib.pyplot as plt


def getPolynomialDataMatrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1,degree + 1):
        X = np.column_stack((X, x ** i))
    return X

def getWeightsForPolynomialFit(x,y,degree):
    X = getPolynomialDataMatrix(x, degree)

    XX = X.transpose().dot(X)
    w = np.linalg.solve(XX, X.transpose().dot(y))
    #w = np.linalg.inv(XX).dot(X.transpose().dot(y))

    return w

plt.figure()
plt.plot(x_test,y_test, 'g')
plt.plot(x_train,y_train, 'bo')

w1 = getWeightsForPolynomialFit(x_train,y_train,1)
Xtest1 = getPolynomialDataMatrix(x_test, 1)
ytest1 = Xtest1.dot(w1)
plt.plot(x_test, ytest1, 'r')

w2 = getWeightsForPolynomialFit(x_train,y_train,2)
Xtest2 = getPolynomialDataMatrix(x_test, 2)
ytest2 = Xtest2.dot(w2)
plt.plot(x_test, ytest2, 'g')

w3 = getWeightsForPolynomialFit(x_train,y_train,3)
Xtest3 = getPolynomialDataMatrix(x_test, 3)
ytest3 = Xtest3.dot(w3)
plt.plot(x_test, ytest3, 'm')

w6 = getWeightsForPolynomialFit(x_train,y_train,6)
Xtest6 = getPolynomialDataMatrix(x_test, 6)
ytest6 = Xtest6.dot(w6)
plt.plot(x_test, ytest6, 'c')

w10 = getWeightsForPolynomialFit(x_train,y_train,10)
Xtest10 = getPolynomialDataMatrix(x_test, 10)
ytest10 = Xtest10.dot(w10)
plt.plot(x_test, ytest10, 'v')

plt.ylim((-200, 200))
plt.legend(('training points', 'ground truth', '$x$', '$x^2$', '$x^3$', '$x^6$', '$x^10$'), loc = 'lower right')
plt.savefig('polynomial.png')
##errors on test dataset Bashir

error1 = y_test-ytest1
SSE1 = error1.dot(error1)

error2 = y_test-ytest2
SSE2 = error2.dot(error2)

error3 = y_test-ytest3
SSE3 = error3.dot(error3)

error6 = y_test-ytest6
SSE6 = error6.dot(error6)

error10 = y_test-ytest10
SSE10 = error10.dot(error10)

SSE1, SSE2, SSE3, SSE6, SSE10

SSEtrain = np.zeros((11,1))
SSEtest = np.zeros((11,1))
MSSEtrain = np.zeros((11,1))
MSSEtest = np.zeros((11,1))

# Feel free to use the functions getWeightsForPolynomialFit and getPolynomialDataMatrix
for i in range(1,12):
    
    Xtrain = getPolynomialDataMatrix(x_train, i) 
    Xtest = getPolynomialDataMatrix(x_test, i)
    
    w = getWeightsForPolynomialFit(x_train, y_train, i)  
    
    MSSEtrain[i - 1] = np.mean((Xtrain.dot(w) - y_train)**2)
    MSSEtest[i - 1] = np.mean((Xtest.dot(w) - y_test)**2)
    
    errortrain = y_train - Xtrain.dot(w) 
    errortest = y_test - Xtest.dot(w)
    SSEtrain[i-1] = errortrain.dot(errortrain)
    SSEtest[i-1] = errortest.dot(errortest)

plt.figure();
plt.semilogy(range(1,12), SSEtrain)
plt.semilogy(range(1,12), SSEtest)
plt.legend(('SSE on training set', 'SSE on test set'))
plt.savefig('polynomial_evaluation.png')
plt.show()