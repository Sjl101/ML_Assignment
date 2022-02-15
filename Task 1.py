import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#converts the file into a pandas dataframe
data_frame = pd.read_csv('pol_regression.csv')
#splits the data depending on their column
x_train = data_frame['x']
y_train = data_frame['y']

def feature_expansion(features, degree):
    X = np.ones(features.shape)
    for i in range(1, degree + 1):
        X = np.column_stack((X, features ** i))

    return X

def pol_regression(x, y, degree):
    X = feature_expansion(x, degree=degree)
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
    
def eval_pol_regression(parameters, x, y, degree):
    MSE = np.square(np.subtract(x, y)).mean()
    RMSE = np.sqrt(MSE)
    return RMSE

features = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

#wd0 = pol_regression(x_train, y_train, degree=0)
#y0 = plot_predictions(weights=wd0, features=features)
#plt.plot(features, y0, color='green')

# Calculates the weights:
wd1 = pol_regression(x_train, y_train, degree=1)
# Calculates predicted:
y1 = plot_predictions(weights=wd1, features=features)
# Plots the data on the graph
plt.plot(features, y1, color='red')

wd2 = pol_regression(x_train, y_train, degree=2)
y2 = plot_predictions(weights=wd2, features=features)
plt.plot(features, y2, color='cyan')

wd3 = pol_regression(x_train, y_train, degree=3)
y3 = plot_predictions(weights=wd3, features=features)
plt.plot(features, y3, color='yellow')

wd6 = pol_regression(x_train, y_train, degree=6)
y6 = plot_predictions(weights=wd6, features=features)
plt.plot(features, y6, color='violet')

wd10 = pol_regression(x_train, y_train, degree=10)
y10 = plot_predictions(weights=wd10, features=features)
plt.plot(features, y10, color='lime')

plt.ylim([-150, 50])
plt.xlim([-4, 4])
plt.scatter(x_train, y_train)
plt.legend(('Degree - 1', "Degree - 2", "Degree - 3", "Degree - 6", "Degree - 10", 'Training Points'), loc = 'lower right')
plt.savefig('pol_regression.png')
plt.show()

train_df, test_df = train_test_split(data_frame, test_size=0.3)

x_train = train_df['x']
y_train = train_df['y']
x_test = test_df['x']
y_test = test_df['y']

ytr1 = plot_predictions(weights=wd1, features=x_train)
RMSE_tr_degree1 = eval_pol_regression(parameters=wd1, x=y_train, y=ytr1, degree=1)
ytr2 = plot_predictions(weights=wd2, features=x_train)
RMSE_tr_degree2 = eval_pol_regression(parameters=wd2, x=y_train, y=ytr2, degree=2)
ytr3 = plot_predictions(weights=wd3, features=x_train)
RMSE_tr_degree3 = eval_pol_regression(parameters=wd3, x=y_train, y=ytr3, degree=3)
ytr6 = plot_predictions(weights=wd6, features=x_train)
RMSE_tr_degree6 = eval_pol_regression(parameters=wd6, x=y_train, y=ytr6, degree=6)
ytr10 = plot_predictions(weights=wd10, features=x_train)
RMSE_tr_degree10 = eval_pol_regression(parameters=wd10, x=y_train, y=ytr10, degree=10)

yte1 = plot_predictions(weights=wd1, features=x_test)
RMSE_te_degree1 = eval_pol_regression(parameters=wd1, x=y_test, y=yte1, degree=1)
yte2 = plot_predictions(weights=wd2, features=x_test)
RMSE_te_degree2 = eval_pol_regression(parameters=wd2, x=y_test, y=yte2, degree=2)
yte3 = plot_predictions(weights=wd3, features=x_test)
RMSE_te_degree3 = eval_pol_regression(parameters=wd3, x=y_test, y=yte3, degree=3)
yte6 = plot_predictions(weights=wd6, features=x_test)
RMSE_te_degree6 = eval_pol_regression(parameters=wd6, x=y_test, y=yte6, degree=6)
yte10 = plot_predictions(weights=wd10, features=x_test)
RMSE_te_degree10 = eval_pol_regression(parameters=wd10, x=y_test, y=yte10, degree=10)

plt.plot(y_test, y_test,  color='blue', linewidth=3)
plt.show()

errors_train = [RMSE_tr_degree1, RMSE_tr_degree2, RMSE_tr_degree3, RMSE_tr_degree6, RMSE_tr_degree10]
degrees = [1, 2, 3, 6, 10]
print(errors_train)

plt.plot(degrees, errors_train, color='red')
plt.savefig('train_errors.png')
plt.show()

errors_test = [RMSE_te_degree1, RMSE_te_degree2, RMSE_te_degree3, RMSE_te_degree6, RMSE_te_degree10]
print(errors_test)
degrees = [1, 2, 3, 6, 10]

plt.plot(degrees, errors_test, color='red')
plt.savefig('test_errors.png')
plt.show()
