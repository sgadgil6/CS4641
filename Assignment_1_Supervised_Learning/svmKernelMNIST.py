
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn import svm
import time
import datetime as dt
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV

training_set = pd.read_csv("./MNISTDataset/mnist_train.csv")
testing_set = pd.read_csv("./MNISTDataset/mnist_test.csv")

x = []
y = []

for row in training_set.iterrows():
    label = row[1][0]
    image = list(row[1][1:])
    image = [round(elem/255.0, 2) for elem in image]
    image = np.array(image)
    x.append(image)
    y.append(label)

x = np.array(x)
y = np.array(y)
# x[x>0] = 1

testing_set_size_arr = []
training_set_size_arr = []
training_time_arr = []
testing_time_arr = []
accuracy_arr = []
cross_val_test_scores = []
cross_val_train_scores = []
kernel_arr = []
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.01, random_state=42)
clfDT = svm.SVC(random_state=42)

#Grid Search is done below to get best C and Gamma Values
parameters = {'gamma': [0.1, 0.001, 0.0001],
                  'C': [1, 10]}


p = GridSearchCV(clfDT, param_grid=parameters, cv=3)
p.fit(x_validation, y_validation)
print("Params=", p.cv_results_['params'])
print("Best Params=",p.best_params_)
print("Scores=", p.cv_results_['mean_test_score'])
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.5, random_state=42)
    print("Kernel =", kernel)
    kernel_arr.append(kernel)
    clfDT = svm.SVC(kernel=kernel, gamma=0.01, C=10)
    start_time = dt.datetime.now()

    crossValEstimator = cross_validate(clfDT, x_train, y_train, cv=5, return_train_score=True)
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
         'Kernel': kernel_arr
}

df = DataFrame(data, columns=[ 'Training Time', 'Testing Time', 'Cross Val Test Score',  'Cross Val Train Score', 'Kernel'])
df.to_csv(r'svmKernelMNIST.csv', index=None, header=True)

