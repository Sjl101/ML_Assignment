import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#splits the data depending on their column
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
x_test = x_test.sort_values(ascending=True)
y_test = y_test.sort_values(ascending=True)

def getPolynomialDataMatrix(features, degree):
    X = np.ones(features.shape)
    for i in range(1, degree + 1):
        X = np.column_stack((X, features ** i))

    return X

def pol_regression(x, y, degree):
    X = getPolynomialDataMatrix(x, degree)

    XX = X.transpose().dot(X)
    w = np.linalg.solve(XX, X.transpose().dot(y))
    #w = np.linalg.inv(XX).dot(X.transpose().dot(y))

    return w
    
def eval_pol_regression(parameters, x, y, degree):  
    MSE = np.square(np.subtract(x, y)).mean()
    RMSE = np.sqrt(MSE)
    return RMSE

features = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]


#wd0 = pol_regression(x_train, y_train, degree=0)
#y0 = plot_predictions(weights=wd0, features=features)
#plt.plot(features, y0, color='green')

# Calculates the weights:
wd1 = pol_regression(x_train,y_train,1)
Xtest1 = getPolynomialDataMatrix(x_test, 1)
# Calculates predicted:
ytest1 = Xtest1.dot(wd1)
# Plots the data on the graph
plt.plot(x_test, ytest1, 'red')

wd2 = pol_regression(x_train,y_train,2)
Xtest2 = getPolynomialDataMatrix(x_test, 2)
ytest2 = Xtest2.dot(wd2)
plt.plot(x_test, ytest2, 'cyan')

wd3 = pol_regression(x_train,y_train,3)
Xtest3 = getPolynomialDataMatrix(x_test, 3)
ytest3 = Xtest3.dot(wd3)
plt.plot(x_test, ytest3, 'yellow')

wd6 = pol_regression(x_train,y_train,6)
Xtest6 = getPolynomialDataMatrix(x_test, 6)
ytest6 = Xtest6.dot(wd6)
plt.plot(x_test, ytest6, 'violet')

wd10 = pol_regression(x_train,y_train,10)
Xtest10 = getPolynomialDataMatrix(x_test, 10)
ytest10 = Xtest10.dot(wd10)
plt.plot(x_test, ytest10, 'lime')

plt.ylim([-150, 50])
plt.xlim([-4, 4])
plt.scatter(x_train, y_train)
plt.legend(('Degree - 1', "Degree - 2", "Degree - 3", "Degree - 6", "Degree - 10", 'Training Points'), loc = 'lower right')
plt.savefig('pol_regression.png')
plt.show()

degrees = [1, 2, 3, 6, 10]

RMSE_degree1 = eval_pol_regression(parameters=wd1, x=y_train, y=ytest1, degree=1)
RMSE_degree2 = eval_pol_regression(parameters=wd2, x=y_train, y=ytest2, degree=2)
RMSE_degree3 = eval_pol_regression(parameters=wd3, x=y_train, y=ytest3, degree=3)
RMSE_degree6 = eval_pol_regression(parameters=wd6, x=y_train, y=ytest6, degree=6)
RMSE_degree10 = eval_pol_regression(parameters=wd10, x=y_train, y=ytest10, degree=10)

# Feel free to use the functions getWeightsForPolynomialFit and getPolynomialDataMatrix


plt.figure();
plt.semilogy(range(1,11), SSEtrain)
plt.semilogy(range(1,11), SSEtest)
plt.legend(('SSE on training set', 'SSE on test set'))
plt.savefig('polynomial_evaluation.png')
plt.show()