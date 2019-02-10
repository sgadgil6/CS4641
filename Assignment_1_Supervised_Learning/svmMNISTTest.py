
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neural_network import MLPClassifier
import time
from sklearn import svm
import datetime as dt
from pandas import DataFrame

training_set = pd.read_csv("./MNISTDataset/mnist_train.csv")
testing_set = pd.read_csv("./MNISTDataset/mnist_test.csv")

x = []
y = []

for row in training_set.iterrows():
    label = row[1][0]
    image = list(row[1][1:])
    image = [round(elem / 255, 2) for elem in image]
    image = np.array(image)
    x.append(image)
    y.append(label)

x = np.array(x)
y = np.array(y)

x_test = []
y_test = []
for row in testing_set.iterrows():
    label = row[1][0]
    image = list(row[1][1:])
    image = [round(elem / 255.0, 2) for elem in image]
    image = np.array(image)
    x_test.append(image)
    y_test.append(label)

x_test = np.array(x_test)
y_test = np.array(y_test)
testing_set_size_arr = []
training_set_size_arr = []
accuracy_arr = []
for i in range(15, 0, -1):
    x_train, _, y_train, _ = train_test_split(x, y, test_size=round(i * 0.06, 2), random_state=42)
    print("Training set size=", len(x_train))
    print("Testing set size=", len(x_test))

    testing_set_size_arr.append(len(x_test))
    training_set_size_arr.append(len(x_train))
    clf = svm.SVC(kernel='rbf', gamma=0.01, C=10)
    start_time = dt.datetime.now()
    clf = clf.fit(x_train, y_train)
    
    y_predicted = clf.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_predicted)
    accuracy_arr.append(accuracy)
    print("Accuracy", accuracy)

    print("---------------------------------------------")
data = {
         'Training Set Size': training_set_size_arr,
         'Testing Set Size': testing_set_size_arr,
         'Accuracy': accuracy_arr
}

df = DataFrame(data, columns=[ 'Training Set Size', 'Testing Set Size', 'Accuracy'])
df.to_csv(r'svmMNISTTest.csv', index=None, header=True)

