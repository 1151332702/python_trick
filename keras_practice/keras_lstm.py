# -*- coding: utf-8 -*-
# @Time    : 2019/3/22 17:30
# @Author  : lilong
# @File    : keras_lstm.py
# @Description: 基于keras的lstm

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

max_features = 1024
model = Sequential(Embedding(input_dim=(128, 20), output_dim=256))
model.add(LSTM(units=128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


