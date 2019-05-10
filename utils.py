import os
import cv2
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import classification_report
import catboost

background_dir = os.listdir('task2img/backgroundimg/')
pores_dir = os.listdir('task2img/poresimg/')


def convert_image(background, pores):
    img_set = []
    temp_list = []
    for i in range(len(background)):
        img = cv2.imread('task2img/backgroundimg/' + background[i], 0)
        for rows in img:
            for pixel in rows:
                temp_list.append(float(pixel))
        img_set.append(temp_list)
        temp_list = []
    back = pd.DataFrame(data=img_set)
    back['label'] = 0
    img_set = []
    temp_list = []
    for i in range(len(pores)):
        img = cv2.imread('task2img/poresimg/' + pores[i], 0)
        for rows in img:
            for pixel in rows:
                temp_list.append(float(pixel))
        img_set.append(temp_list)
        temp_list = []
    pore = pd.DataFrame(data=img_set)
    pore['label'] = 1
    data = back.append(pore, ignore_index=True)
    x = data.drop(columns='label', axis=1)
    y = data['label']
    x.to_csv("x.csv", index=False, header=False)
    y.to_csv("y.csv", index=False)


# convert_image(background_dir,pores_dir)

def svm_sample():
    x = pd.read_csv("x.csv")
    y = np.ravel(pd.read_csv("y.csv"))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train)
    x_test = scale.transform(x_test)
    clf = SVC()
    clf.fit(x_train, y_train)
    joblib.dump(clf, 'clf.pkl')
    y_predict = clf.predict(x_test)

    print("Accuracy: {}".format(metrics.accuracy_score(y_test, y_predict)))
    print(classification_report(y_test, y_predict))


# svm_sample()

def catboost_sample():
    x = pd.read_csv("x.csv")
    y = np.ravel(pd.read_csv("y.csv"))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train)
    x_test = scale.transform(x_test)
    clf = catboost.CatBoostClassifier(depth=7, learning_rate=0.03, l2_leaf_reg=6, loss_function='CrossEntropy',
                                      border_count=32,
                                      iterations=500)
    clf.fit(x_train, y_train)
    # clf = catboost.CatBoostClassifier()
    # clf.load_model('catboost.model')
    clf.save_model('catboost.model')
    y_predict = clf.predict(x_test)
    print("Accuracy: {}".format(metrics.accuracy_score(y_test, y_predict)))
    print(classification_report(y_test, y_predict))


catboost_sample()
