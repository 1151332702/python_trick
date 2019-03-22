# -*- coding: utf-8 -*-
# @Time    : 2019/3/18 18:44
# @Author  : lilong
# @File    : xgboost_Model.py
# @Description:
import xgboost as xgb
from sklearn.datasets import load_iris

if __name__ == '__main__':
    data_train = xgb.DMatrix(r'E:\pyProject\python_trick\xgboost_practice\data\14.agaricus_train.txt')
    data_test = xgb.DMatrix(r'E:\pyProject\python_trick\xgboost_practice\data\14.agaricus_test.txt')
    iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target
    # 设置参数
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}  # logitraw
    bst = xgb.XGBModel(objective="reg:linear",
                       booster='gbtree',
                       max_depth=3,
                       learning_rate=1,
                       n_estimators=4
                       )
    bst.fit(X, y)
    y_pred = bst.predict(X)
    y = data_test.get_label()

    error = sum(y != (y_pred > 0.5))
    err_rate = float(error) / len(y_pred)

    print('错误数据%d' % error)
    print('错误率%.5f%%' % (err_rate * 100))
