# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 19:04
# @Author  : lilong
# @File    : keras_intro.py
# @Description:

from keras import Sequential
from keras.layers import Dense, Activation
import keras
import numpy as np
from keras.utils import np_utils

X = np.random.random((1000,100))
_y = np.random.randint(2,size=(1000,1))
y = np_utils.to_categorical(_y, 2)
# 把标签转换为one-hot编码
one_hot_labels = keras.utils.to_categorical(_y, num_classes=2)

model = Sequential()
# 构建网络层数  第一层需要指定输入形状 后续不需要指定
# 首层形状指定  input_dim=100 等价于 input_shape=(100,)
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=2, activation='softmax'))
# 编译模型
# loss: categorical_crossentropy  mse binary_crossentropy
# optimizer: rmsprop  adagrad sgd
# metrics: accuracy 自定义评价函数的时候传递函数名即可 不需要加括号
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# 可以进一步配置
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
# 训练模型
model.fit(X, y, batch_size=32)

# loss  多分类问题： categorical_crossentropy
#       二分类问题： binary_crossentropy
#       回归问题： mse
# 评估模型
loss_and_metrics = model.evaluate(X, y, batch_size=128)
print(loss_and_metrics)
