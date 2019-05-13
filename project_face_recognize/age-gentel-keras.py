# -*- coding: utf-8 -*-
# @Time    : 2019/4/2 17:50
# @Author  : lilong
# @File    : age-gentel-keras.py
# @Description:
import glob
import os
from random import shuffle

import keras
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model
# 变量
age_table = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
sex_table = ['f', 'm']  # f:女; m:男
batch_size = 16
img_w = 816
img_h = 816
channels = 3
# AGE==True 训练年龄模型，False,训练性别模型
AGE = False
if AGE == True:
    lables_size = len(age_table)  # 年龄
else:
    lables_size = len(sex_table)  # 性别

face_set_fold = r'E:\data\Keras_age_gender-detect\AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification'
model_path = r'd:\model.h5'

fold_0_data = os.path.join(face_set_fold, 'fold_0_data.txt')
# fold_1_data = os.path.join(face_set_fold, 'fold_1_data.txt')
# fold_2_data = os.path.join(face_set_fold, 'fold_2_data.txt')
# fold_3_data = os.path.join(face_set_fold, 'fold_3_data.txt')
# fold_4_data = os.path.join(face_set_fold, 'fold_4_data.txt')
face_image_set = os.path.join(face_set_fold, 'aligned')

def parse_data(fold_x_data):
    data_set = []
    with open(fold_x_data, 'r') as f:
        line_one = True
        for line in f:
            tmp = []
            if line_one == True:
                line_one = False
                continue
            print(r"line.split('\t')[0]:", line.split('\t')[0])
            print(r"line.split('\t')[1]:", line.split('\t')[1])
            print(r"line.split('\t')[2]:", line.split('\t')[2])
            print(r"line.split('\t')[3]:", line.split('\t')[3])
            print(r"line.split('\t')[4]:", line.split('\t')[4])
            tmp.append(line.split('\t')[0])  #文件名
            tmp.append(line.split('\t')[1])  #图片名称
            tmp.append(line.split('\t')[3])
            tmp.append(line.split('\t')[4])  #性别

            file_path = os.path.join(face_image_set, tmp[0])
            if os.path.exists(file_path):
                filenames = glob.glob(file_path + "/*.jpg")
                for filename in filenames:
                    if tmp[1] in filename:
                        break
                if AGE == True:
                    if tmp[2] in age_table:
                        data_set.append([filename, np.float32(age_table.index(tmp[2]))])
                else:
                    if tmp[3] in sex_table:
                        data_set.append([filename, np.float32(sex_table.index(tmp[3]))])
    shuffle(data_set)
    return np.array(data_set)

def get_image_data(path, img_w, img_h, channels):
    img = load_img(path, target_size=(img_w, img_h))
    img = img_to_array(img, dtype=np.float32) / 255
    img = np.reshape(img, (img_w, img_h, channels))
    return img

def batch_generator(data_set, batch_size, img_w, img_h, channels):
    '''
    param：
        data_set：数据集[[file name, label], [file name, label]] np.array
        batch_size:批次
        img_w:图片宽
        img_h:图片高
        channels: channels
    返回:
        generator，x: 获取的批次图片 y: 获取的图片对应的标签
    '''
    while 1:
        for i in range(0, len(data_set), batch_size):
            batch_data = data_set[i:i+batch_size]
            x = batch_data[:, 0].ravel()
            y = batch_data[:, 1].ravel().astype(np.float32)
            y = keras.utils.to_categorical(y, num_classes=lables_size)
            x_list = []
            for path in x:
                img = get_image_data(path, img_w, img_h, channels)
                x_list.append(img)
            yield(np.array(x_list), y)

def train(data_set, steps_per_epoch):
    model = Sequential()
    model.add(Conv2D(8, (3, 3),
                     activation='relu',
                     input_shape=(816, 816, 3)))
    model.add(Conv2D(8, (3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.5))

    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(lables_size, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    model.fit_generator(batch_generator(data_set, batch_size, img_w, img_h, channels),
                        steps_per_epoch=steps_per_epoch,
                        epochs=3,
                        verbose=1)
    # score = model.evaluate(x_test, y_test)
    model.save(model_path)

def predict(picture_path, model_path):
    image_data = get_image_data(picture_path, img_w, img_h, channels)
    model = load_model(model_path)
    if AGE:
        pred = age_table[model.predict(image_data.reshape(-1, 816, 816, 3)).ravel().argmax()]
        print('预测年龄区间为: %s' % pred)
    else:
        pred = sex_table[model.predict(image_data.reshape(-1, 816, 816, 3)).ravel().argmax()]
        print('预测性别为: %s' % pred)
    return pred
if __name__ == '__main__':
    # 加载数据
    data_set = parse_data(fold_0_data)
    steps_per_epoch = (len(data_set) // batch_size) + 1

    # 训练模型
    train(data_set, steps_per_epoch)

    # 预测
    # test_picture_path = r'E:\data\Keras_age_gender-detect\AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification\aligned\7153718@N04\landmark_aligned_face.2282.11598106935_115e366e57_o.jpg'
    # predict(picture_path=test_picture_path, model_path=model_path)
