#!/usr/bin/env python
# coding: utf-8



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Classifiers.OS_CNN.OS_CNN_easy_use import OS_CNN_easy_use
import numpy as np


import pandas as pd
df = pd.read_csv(r'filepath', index_col=0)
X = df[df.columns[2:]].values
y = df['class'].values


X_train,X_test,y_train,y_test= train_test_split(X, y,test_size=0.2)



y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)



model = OS_CNN_easy_use(
        Result_log_folder = (r"C:\Users\saath\OneDrive\Desktop\Result_log_folder"), # the Result_log_folder
        dataset_name = (r"Dataset_log"),           # dataset_name for log folder creation under Result_log_folder
        device = "cuda:0",                     # the Gpu you want to use
        max_epoch =100  
       )
 

model.fit(X_train, y_train, X_test, y_test)
y_predict = model.predict(X_test)
print('correct:',y_test)
print('predict:',y_predict)
acc = accuracy_score(y_predict, y_test)
print(acc)




