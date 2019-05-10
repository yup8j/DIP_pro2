import argparse
import cv2
import os
import numpy as np
import pandas as pd
from imutils import paths
from sklearn.externals import joblib
import cv2
import catboost
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import classification_report


my_dict = {1: "pores", 0: "background"}

clf = catboost.CatBoostClassifier()
clf.load_model('catboost.model')
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--testing", required=True,
                help="path to the tesitng images")
args = vars(ap.parse_args())
scale = StandardScaler()
for imagePath in paths.list_images(args["testing"]):
    temp_list = []
    image = cv2.imread(imagePath, 0)
    sx = image.shape
    for rows in image:
        for pixel in rows:
            temp_list.append(float(pixel))
    temp_list = pd.DataFrame(data=temp_list)
    temp_list = scale.fit_transform(temp_list)
    temp_list = np.reshape(temp_list, (1, -1))
    y = clf.predict(temp_list)
    print(imagePath, my_dict[y[0]])
