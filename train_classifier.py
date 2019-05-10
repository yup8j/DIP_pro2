import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings

import os
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)
'''
l  C：C-SVC的惩罚参数C 默认值是1.0

C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。

l  kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 

  　　0 – 线性：u'v

 　　 1 – 多项式：(gamma*u'*v + coef0)^degree

  　　2 – RBF函数：exp(-gamma|u-v|^2)

  　　3 –sigmoid：tanh(gamma*u'*v + coef0)

l  degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。

l  gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features

l  coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。

l  probability ：是否采用概率估计？.默认为False

l  shrinking ：是否采用shrinking heuristic方法，默认为true

l  tol ：停止训练的误差值大小，默认为1e-3

l  cache_size ：核函数cache缓存大小，默认为200

l  class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)

l  verbose ：允许冗余输出？

l  max_iter ：最大迭代次数。-1为无限制。

l  decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3

l  random_state ：数据洗牌时的种子值，int值

主要调节的参数有：C、kernel、degree、gamma、coef0。
'''
x = pd.read_csv("x.csv")
y = pd.read_csv("y.csv")
y = np.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)
#
# param_grid = {"C": [0.01, 0.1, 1, 10]}
#
# clf = GridSearchCV(SVC(gamma='auto'), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1,verbose=100)
# clf.fit(X_train, y_train)
#
# print("Best parameters set found on development set:\n {}".format(clf.best_params_))

import catboost

# param_grid = {"depth": [4, 5, 6, 7]}

# param_grid_1 = {"l2_leaf_reg": [2, 3, 4, 5, 6]}
param_grid = {"learning_rate": [0.01, 0.03, 0.1]}
clf = GridSearchCV(
    catboost.CatBoostClassifier(learning_rate=0.03, l2_leaf_reg=3, loss_function='Logloss', border_count=32,
                                iterations=500), param_grid=param_grid, cv=10, n_jobs=-1, verbose=10,
    scoring='accuracy')
clf.fit(X_train, y_train)
print("Best parameters set found on development set:\n {}".format(clf.best_params_))
