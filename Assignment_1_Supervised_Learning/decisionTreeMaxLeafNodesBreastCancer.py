
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn import tree
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
max_leaf_nodes_arr = []

for i in range(100, 1000, 100):
    max_leaf_nodes_arr.append(i)
    print("Max depth =", i)
    clfDT = tree.DecisionTreeClassifier(max_leaf_nodes=i)
    start_time = dt.datetime.now()
    # clfDT = clfDT.fit(x_train, y_train)
    # end_time = dt.datetime.now()
    # elapsed_time= end_time - start_time
    crossValEstimator = cross_validate(clfDT, x, y, cv=5, return_train_score=True)
    print('Time to learn {}'.format(str(crossValEstimator['fit_time'])))
    training_time_arr.append(np.average(crossValEstimator['fit_time']))

    # start_time = dt.datetime.now()
    # y_predicted = clfDT.predict(x_test)
    # end_time = dt.datetime.now()
    # elapsed_time = end_time - start_time
    print('Time to predict {}'.format(str(crossValEstimator['score_time'])))
    testing_time_arr.append(np.average(crossValEstimator['score_time']))

    # print(y_predicted[0:20], ".....")
    # print(y_test[0:20], ".....")

    # y_predicted =
    # accuracy = metrics.accuracy_score(y_test, y_predicted)
    #print(metrics.confusion_matrix(y_validation, y_predicted))
    # print("F1 score = ", metrics.f1_score(y_test, y_predicted))
    # f1_score_arr.append(metrics.f1_score(y_test, y_predicted))

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
         'Max Leaf Nodes': max_leaf_nodes_arr
}

df = DataFrame(data, columns=[ 'Training Time', 'Testing Time', 'Cross Val Test Score',  'Cross Val Train Score', 'Max Leaf Nodes'])
df.to_csv(r'decisionTreeMaxLeafNodesBreastCancer.csv', index=None, header=True)

