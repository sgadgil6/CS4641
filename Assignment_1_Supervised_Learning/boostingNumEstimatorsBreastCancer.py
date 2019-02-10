
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.ensemble import AdaBoostClassifier
import time
import datetime as dt
from pandas import DataFrame

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
num_estimators_arr = []

for num_estimators in range(10, 350, 30):
    num_estimators_arr.append(num_estimators)
    clf = AdaBoostClassifier(n_estimators=num_estimators)
    start_time = dt.datetime.now()

    crossValEstimator = cross_validate(clf, x, y, cv=5, return_train_score=True)
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
df.to_csv(r'boostingNumEstimatorsBreastCancer.csv', index=None, header=True)

