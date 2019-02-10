
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
from sklearn.tree import DecisionTreeClassifier
import datetime as dt
from pandas import DataFrame

train_set = pd.read_csv("./breastCancerDataset/breastCancer_train.csv")
test_set = pd.read_csv("./breastCancerDataset/breastCancer_test.csv")

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

x_test = []
y_test = []
for row in test_set.iterrows():
    diagnosis = row[1][-1]
    features = list(row[1][0:-2])
    features = np.array(features)
    x_test.append(features)
    y_test.append(diagnosis)
testing_set_size_arr = []
training_set_size_arr = []
accuracy_arr = []
for i in range(18, 0, -1):
    x_train, _, y_train, _ = train_test_split(x, y, test_size=round(i * 0.05, 2), random_state=42)
    print("Training set size=", len(x_train))
    print("Testing set size=", len(x_test))

    testing_set_size_arr.append(len(x_test))
    training_set_size_arr.append(len(x_train))
    clf = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=300)
    start_time = dt.datetime.now()
    clf = clf.fit(x_train, y_train)

    y_predicted = clf.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_predicted)
    accuracy_arr.append(accuracy)
    print("Accuracy", accuracy)
    print(metrics.confusion_matrix(y_test, y_predicted))
    print("F1 score = ", metrics.f1_score(y_test, y_predicted))
    print("Precision = ", metrics.precision_score(y_test, y_predicted))
    print("Recall = ", metrics.recall_score(y_test, y_predicted))
    # f1_score_arr.append(metrics.f1_score(y_test, y_predicted))
    print("---------------------------------------------")
data = {
         'Training Set Size': training_set_size_arr,
         'Testing Set Size': testing_set_size_arr,
         'Accuracy': accuracy_arr
}

df = DataFrame(data, columns=[ 'Training Set Size', 'Testing Set Size', 'Accuracy'])
df.to_csv(r'decisionTreeBreastCancerTest.csv', index=None, header=True)

