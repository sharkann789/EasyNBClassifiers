# loading libraries
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error

from NBClassifier import *

data = load_iris()
X = data['data']
y = data['target']
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


print(shape(X_train))
print(shape(y_train))


model = NBClassifier(X_train, y_train)
# you could train here but hey lmao

#pred = model.predict(X_test[0])

testvals = []
corr = 0
for x in range(0, len(X_test)):
    val = model.predict(X_test[x])
    if (val == y_test[x]):
        corr += 1

    testvals.append(val)

print(testvals)
print(y_test)
print( corr / len(y_test))
print("MSE loss: " + str(mean_squared_error(y_test, testvals)))