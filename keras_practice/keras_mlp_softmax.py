# -*- coding: utf-8 -*-
# @Time    : 2019/3/20 16:26
# @Author  : lilong
# @File    : keras_mlp_softmax.py
# @Description: 基于mlp的softmax多分类 mnist数据集

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
import numpy as np
from keras.datasets import mnist
from keras.utils.vis_utils import plot_model

batch_size = 128
num_classes = 10
epochs = 20
# the data, split between train and test sets
mnist_data = np.load('../data/mnist.npz')
x_train, y_train, x_test, y_test = mnist_data['x_train'], mnist_data['y_train'], mnist_data['x_test'], mnist_data['y_test']
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(dtype='float32')
x_test = x_test.astype(dtype='float32')
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# one hot encode
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=784))
model.add(Dropout(0.8))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=RMSprop(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
# x_pred = np.random.random((1, 20))
# pred = model.predict(x_pred)
# print(pred)
print('test loss', score[0])
print('test acc', score[1])