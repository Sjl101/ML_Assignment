from cgi import test
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import KFold
from keras.models import Model
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn import datasets

df = pd.read_csv('Task 3/HIV_RVG.csv')
print(df)

def display_boxplot(data, column):
    data.boxplot(by='Participant Condition', column=column)
    test = f"""boxplot_{column}.png"""
    plt.savefig(test)
    plt.show()

def statistical_summary(data):
    #mean values, standard deviations, min/max values
    data = data.drop(columns=['Participant Condition'])
    min = {}
    max = {}
    mean = {}
    min['summary'] = ['min']
    max['summary'] = ['max']
    mean['summary'] = ['mean']
    for i in data.columns:
        min[i] = [data[i].min()]
    for i in data.columns:
        max[i] = [data[i].max()]
    for i in data.columns:
        mean[i] = [data[i].mean()]
    mindf = pd.DataFrame(min)
    maxdf = pd.DataFrame(max)
    meandf = pd.DataFrame(mean)
    sdf = pd.concat([mindf, maxdf, meandf])
    print(sdf)
    return mindf, maxdf, meandf

def split_data(data):
    print('prep')
    raw = data['Participant Condition'].to_numpy()  
    le = preprocessing.LabelEncoder()
    le.fit(raw)
    labels = le.transform(raw)
    data['Labels'] = labels
    traindf, testdf = train_test_split(data, test_size=0.1)
    return traindf, testdf

def ann10Fold(df, neurons, k, epochs):
    print('prep')
    raw = df['Participant Condition'].to_numpy()  
    l = preprocessing.LabelEncoder()
    l.fit(raw)
    labels = l.transform(raw)
    df['Labels'] = labels
    traindf, testdf = train_test_split(df, test_size=0.1)
    # Create the neural network:
    model = keras.models.Sequential([keras.layers.Flatten(input_shape=[5]),keras.layers.Dense(neurons, activation='sigmoid'),keras.layers.Dense(neurons, activation='sigmoid'),keras.layers.Dense(1, activation='sigmoid')])
    # Complie the model:
    model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
    k = KFold(n_splits=k, random_state=None)
    acc_score = []
    f = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    f = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    acc_score = np.zeros(shape=(len(f), 0))  
    for train_index, test_index in k.split(df):
        train = df.iloc[train_index]
        test = df.iloc[test_index]
        # Select the values to create X:
        train_x = train[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
        test_x = test[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
        # Select the values to create Y:
        train_y = train[['Labels']].to_numpy()
        test_y = test[['Labels']].to_numpy()
        # Reshape the Y values:
        train_y = np.reshape(train_y, (-1, len(train_y)))
        train_y = train_y[0]   
        # Train the model here:
        history = model.fit(train_x, train_y, epochs=epochs) 
        y_pred = model.predict(test_x)
        print(y_pred)
        print(test_y)
        ypred1d = y_pred.flatten()
        acc = accuracy_score(ypred1d.round(), test_y)
        acc_score = np.array([])
        acc_score = np.append(acc_score, acc)
        
    print(acc_score)
    print(f)
    print(k)
    
    return acc_score

def ann(train, test, epochs):
    # Select values to create X and Y: 
    tr_y = train[['Labels']].to_numpy()
    te_y = train[['Labels']].to_numpy()
    tr_x = train[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
    te_x = test[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
    
    # Reshapes the Y values
    tr_y = np.reshape(tr_y, (-1, len(tr_y)))
    te_y = np.reshape(te_y, (-1, len(te_y)))
    tr_y = tr_y[0]
    te_y = te_y[0]
    # Create neural network model
    model = keras.models.Sequential([keras.layers.Flatten(input_shape=[5]),keras.layers.Dense(500, activation='sigmoid'),keras.layers.Dense(500, activation='sigmoid'),keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
    # Train the model:
    h = model.fit(tr_x, tr_y, epochs=epochs)
    # Present any metrics that it produces:
    hdf = pd.DataFrame(h.history).plot(figsize=(8, 5))
    test = f"""accuracy_graph_epoch_{epochs}.png"""
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(test)
    plt.show()


k = 10
display_boxplot(df, 'Alpha')
display_boxplot(df, 'Beta')
mindf, maxdf, meandf = statistical_summary(df)
traindata, testdata = split_data(df)
ann(traindata, testdata, 64)
ann(traindata, testdata, 128)
ann(traindata, testdata, 256)
listdata = np.array([])
acc_scores_1000 = ann10Fold(df, 1000, 10, 30)
listdata = np.append(listdata, acc_scores_1000)
acc_scores_500 = ann10Fold(df, 500, 10, 30)
listdata = np.append(listdata, acc_scores_500)
acc_scores_50 = ann10Fold(df, 50, 10, 30)
listdata = np.append(listdata, acc_scores_50)
print(listdata)