
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
import time
import datetime as dt
from pandas import DataFrame

data_set = pd.read_csv("./breastCancerDataset/Breast_cancer_data.csv")

x = []
y = []

for row in data_set.iterrows():
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
metric_arr = []

for distMetric in {'euclidean', 'manhattan', 'chebyshev', 'minkowski'}:
    metric_arr.append(distMetric)
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2,  random_state=42)
    print("Training set size=", len(x_train))
    print("Testing set size=", len(x_validation))
    print("Distance Metric =", distMetric)

    testing_set_size_arr.append(len(x_validation))
    training_set_size_arr.append(len(x_train))
    clf = KNeighborsClassifier(metric=distMetric, n_jobs=-1)
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
    # print(accuracy)
    # accuracy_arr.append(str(accuracy))
    print("---------------------------------------------")
data = {
         'Training Time': training_time_arr,
         'Testing Time': testing_time_arr,
         'Cross Val Test Score': cross_val_test_scores,
         'Cross Val Train Score': cross_val_train_scores,
         'Distance Metric': metric_arr
}

df = DataFrame(data, columns=[ 'Training Time', 'Testing Time', 'Cross Val Test Score',  'Cross Val Train Score', 'Distance Metric'])
df.to_csv(r'knnDistMetricBreastCancer.csv', index=None, header=True)

