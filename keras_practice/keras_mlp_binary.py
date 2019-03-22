# -*- coding: utf-8 -*-
# @Time    : 2019/3/21 20:43
# @Author  : lilong
# @File    : keras_mlp_binary.py
# @Description:  多层感知机的二分类问题

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# 产生虚拟数据
X_train = np.random.random((1000, 20))
y_train = np.random.randint(low=0, high=2, size=(1000, 1))
# print(X_train)
# print(y_train)
X_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(units=64, input_dim=20, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=200, epochs=10);
score = model.evaluate(X_test, y_test)
print(score)
y_pred = model.predict(X_test)
print(y_pred)

