# -*- coding: utf-8 -*-
# @Time    : 2019/3/21 8:44
# @Author  : lilong
# @File    : predict_boston.py
# @Description:  预测波士顿房价
import pandas as pd
from keras.layers import Dense, Dropout, BatchNormalization
from keras import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


data = pd.read_csv(r'E:\pyProject\DataAnalyzerFrame\test\data\reg\train.csv').values
scaler = StandardScaler()
X = data[:, :-1]
y = data[:, -1]
X_train = X[:400]
X_test = X[400:]

y_train = y[:400]
y_test = y[400:]

model = Sequential()
model.add(Dropout(0.8))
model.add(Dense(units=64, activation='relu', input_dim=14))
model.add(BatchNormalization())
model.add(Dense(units=1, activation='relu'))
model.compile(loss='mse', optimizer="adam")

model.fit(X_train, y_train, batch_size=64, epochs=3)

pred = model.predict(X_test)

print(mean_absolute_error(y_test, pred))

