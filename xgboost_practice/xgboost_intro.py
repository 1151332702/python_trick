# -*- coding: utf-8 -*-
# @Time    : 2019/3/15 15:36
# @Author  : lilong
# @File    : xgboost_intro.py
# @Description: xgboost简介
# 多分类问题指定 objective为'multi:softmax'
import numpy as np
import xgboost as xgb
xgb.XGBModel()

if __name__ == '__main__':
    data_train = xgb.DMatrix(r'E:\pyProject\python_trick\xgboost_practice\data\14.agaricus_train.txt')
    data_test = xgb.DMatrix(r'E:\pyProject\python_trick\xgboost_practice\data\14.agaricus_test.txt')
    print(data_train)
    print(data_test)
    # 设置参数
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}  # logitraw
    # xgb.XGBClassifier()
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    bst = xgb.train(param, data_train, num_boost_round=3, evals=watchlist)
    # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate)
    y_pred = bst.predict(data_test)
    y = data_test.get_label()

    error = sum(y != (y_pred > 0.5))
    err_rate = float(error) / len(y_pred)

    print('错误数据%d' % error)
    print('错误率%.5f%%' % (err_rate * 100))