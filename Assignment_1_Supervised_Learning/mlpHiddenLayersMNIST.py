
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neural_network import MLPClassifier
import time
import datetime as dt
from pandas import DataFrame

training_set = pd.read_csv("./MNISTDataset/mnist_train.csv")
testing_set = pd.read_csv("./MNISTDataset/mnist_test.csv")

x = []
y = []

for row in training_set.iterrows():
    label = row[1][0]
    image = list(row[1][1:])
    image = np.array(image)
    x.append(image)
    y.append(label)

x_test = []
y_test = []
for row in testing_set.iterrows():
    label = row[1][0]
    image = list(row[1][1:])
    image = np.array(image)
    x_test.append(image)
    y_test.append(label)

x_test = np.array(x_test)
y_test = np.array(y_test)
testing_set_size_arr = []
training_set_size_arr = []
training_time_arr = []
testing_time_arr = []
accuracy_arr = []
cross_val_test_scores = []
cross_val_train_scores = []
num_hidden_layers = []

for hidden_layers in range(1,15,2):
    num_hidden_layers.append(hidden_layers)
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.5, random_state=42)
    print("Training set size=", len(x_train))
    print("Testing set size=", len(x_test))
    print("Num hidden layers =", hidden_layers)

    testing_set_size_arr.append(len(x_test))
    training_set_size_arr.append(len(x_train))
    clf = MLPClassifier(hidden_layer_sizes=[100]*hidden_layers)
    start_time = dt.datetime.now()
    # clfDT = clfDT.fit(x_train, y_train)
    # end_time = dt.datetime.now()
    # elapsed_time= end_time - start_time
    crossValEstimator = cross_validate(clf, x_train, y_train, cv=5, return_train_score=True)
    print('Time to learn {}'.format(str(crossValEstimator['fit_time'])))
    training_time_arr.append(np.average(crossValEstimator['fit_time']))


    print('Time to predict {}'.format(str(crossValEstimator['score_time'])))
    testing_time_arr.append(np.average(crossValEstimator['score_time']))



    print("Cross Val Test Score = ", crossValEstimator['test_score'])
    print("Cross Val Train Score = ", crossValEstimator['train_score'])
    cross_val_test_scores.append( np.average(crossValEstimator['test_score']))
    cross_val_train_scores.append(np.average(crossValEstimator['train_score']))
    # print(accuracy)
    # accuracy_arr.append(str(accuracy))
    print("---------------------------------------------")
data = {
         'Training Time': training_time_arr,
         'Testing Time': testing_time_arr,
         'Cross Val Test Score': cross_val_test_scores,
         'Cross Val Train Score': cross_val_train_scores,
         'Num Hidden Layers': num_hidden_layers}

df = DataFrame(data, columns=[ 'Training Time', 'Testing Time', 'Cross Val Test Score',  'Cross Val Train Score', 'Num Hidden Layers'])
df.to_csv(r'mlpHiddenLayersMNIST.csv', index=None, header=True)

