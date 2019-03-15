# -*- coding: utf-8 -*-
# @Time    : 2019/3/15 15:36
# @Author  : lilong
# @File    : xgboost_intro.py
# @Description: xgboost简介
import numpy as np
import xgboost as xgb

if __name__ == '__main__':
    data_train = xgb.DMatrix(r'E:\pyProject\python_trick\xgboost_practice\data\14.agaricus_train.txt')
    data_test = xgb.DMatrix(r'E:\pyProject\python_trick\xgboost_practice\data\14.agaricus_test.txt')
    print(data_train)
    print(data_test)
    xgb.XGBClassifier()