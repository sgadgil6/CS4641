
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.ensemble import AdaBoostClassifier
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
num_estimators_arr = []

for num_estimators in range(10, 200, 30):
    num_estimators_arr.append(num_estimators)
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.5, random_state=42)
    print("Training set size=", len(x_train))
    print("Testing set size=", len(x_test))
    print("Num Estimators =", num_estimators)

    testing_set_size_arr.append(len(x_test))
    training_set_size_arr.append(len(x_train))
    clf = AdaBoostClassifier(n_estimators=num_estimators)
    start_time = dt.datetime.now()

    crossValEstimator = cross_validate(clf, x_train, y_train, cv=5, return_train_score=True)
    print('Time to learn {}'.format(str(crossValEstimator['fit_time'])))
    training_time_arr.append(np.average(crossValEstimator['fit_time']))

    print('Time to predict {}'.format(str(crossValEstimator['score_time'])))
    testing_time_arr.append(np.average(crossValEstimator['score_time']))

    print("Cross Val Test Score = ", crossValEstimator['test_score'])
    print("Cross Val Train Score = ", crossValEstimator['train_score'])
    cross_val_test_scores.append( np.average(crossValEstimator['test_score']))
    cross_val_train_scores.append(np.average(crossValEstimator['train_score']))

    print("---------------------------------------------")
data = {
         'Training Time': training_time_arr,
         'Testing Time': testing_time_arr,
         'Cross Val Test Score': cross_val_test_scores,
         'Cross Val Train Score': cross_val_train_scores,
         'Number of Estimators': num_estimators_arr
}

df = DataFrame(data, columns=[ 'Training Time', 'Testing Time', 'Cross Val Test Score',  'Cross Val Train Score', 'Number of Estimators'])
df.to_csv(r'boostingNumEstimatorsMNIST.csv', index=None, header=True)

