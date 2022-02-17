from cgi import test
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

df = pd.read_csv('Task 3/HIV_RVG.csv')
print(df)

def display_boxplot(data, column):
    data.boxplot(by='Participant Condition', column=column)
    plt.show()

def statistical_summary(data):
    #mean values, standard deviations, min/max values
    data = data.drop(columns=['Participant Condition'])
    columns = ["Image number","Bifurcation number","Artery (1)/ Vein (2)","Alpha","Beta","Lambda","Lambda1","Lambda2"]
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
    summarydf = pd.concat([mindf, maxdf, meandf])
    print(summarydf)

display_boxplot(df, 'Alpha')
display_boxplot(df, 'Beta')
statistical_summary(df)