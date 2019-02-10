
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neural_network import MLPClassifier
import time
import datetime as dt
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

train_set = pd.read_csv("./breastCancerDataset/breastCancer_train.csv")


x = []
y = []

for row in train_set.iterrows():
    diagnosis = row[1][-1]
    features = list(row[1][0:-2])
    features = np.array(features)
    x.append(features)
    y.append(diagnosis)

x = np.array(x)
y = np.array(y)

testing_set_size_arr = []
training_set_size_arr = []
training_time_arr = []
testing_time_arr = []
accuracy_arr = []
cross_val_test_scores = []
cross_val_train_scores = []
num_neurons_arr = []

for num_neurons in range(5,150,10):
    num_neurons_arr.append(num_neurons)

    clf = MLPClassifier(hidden_layer_sizes=(num_neurons, num_neurons, num_neurons, num_neurons))

    crossValEstimator = cross_validate(clf, x, y, cv=5, return_train_score=True)
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
         'Num Neurons': num_neurons_arr}

df = DataFrame(data, columns=[ 'Training Time', 'Testing Time', 'Cross Val Test Score',  'Cross Val Train Score', 'Num Neurons'])
df.to_csv(r'mlpNumNeuronsBreastCancer.csv', index=None, header=True)

