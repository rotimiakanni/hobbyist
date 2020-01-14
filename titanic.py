# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:10:26 2019

@author: ROTIMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
#import csv

train_csv = pd.read_csv('trainTitanic.csv')
final_csv = pd.read_csv('testTitanic.csv')


#sex_transform = []
#for sex in dataset.Sex:
#    if sex == 'male':
#        sex_transform.append(1)
#    else:
#        sex_transform.append(2)
#dataset['sex_transform'] = sex_transform
#numeric_features = dataset.select_dtypes(include=[np.number])
#numeric_features = numeric_features.drop(['PassengerId'], axis=1)
#numeric_features = numeric_features.select_dtypes(include=[np.number]).interpolate().dropna()
#numeric_features = numeric_features[numeric_features['Fare']<300]
##numeric_features = numeric_features[numeric_features['Age']<70]
#Y = numeric_features['Survived']
#X = numeric_features.drop('Survived', axis=1)
#
#
##plt.plot(X.Age[Y==1], 'ro')
###plt.plot(X.Age[Y==0], X.Pclass[Y==0], 'go')
##plt.xlabel('age')
##plt.ylabel('fare')
#
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.02, random_state=42)
#
#clf = clf = LogisticRegression(solver='lbfgs')
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
##plt.plot(X_train["sepal.length"], X_train["sepal.width"])
#print(accuracy_score(y_test, y_pred))
#
#test_data = test_dataset
#test_sex_transform = []
#sex_transform = []
#for sex in test_data.Sex:
#    if sex == 'male':
#        test_sex_transform.append(1)
#    else:
#        test_sex_transform.append(2)
#        
#test_data['sex_transform'] = test_sex_transform
#test_data = test_data.select_dtypes(include=[np.number]).interpolate().drop('PassengerId', axis=1)
#print(test_data.head())
#y_test_pred = clf.predict(test_data)
#print(len(y_test_pred))
#
#append_list = [['PassengerId', 'Survived']]
#
#for x,y in zip(list(test_dataset.PassengerId), list(y_test_pred)):
#    append_list.append([x, y])
#
#print(append_list[:10])
#
#with open('submit.csv', 'w') as csvFile:
#    writer = csv.writer(csvFile)
#    writer.writerows(append_list)
#
#csvFile.close()









import re
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def get_simplified_title(csv):
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return ""
    title = csv['Name'].apply(get_title)
    sim_title = title.replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    sim_title = sim_title.replace('Mlle', 'Miss')
    sim_title = sim_title.replace('Ms', 'Miss')
    sim_title = sim_title.replace('Mme', 'Mrs')
    return sim_title

train_csv['FamilyCount'] = train_csv['SibSp'] + train_csv['Parch'] + 1

train_csv['SimplifiedTitle'] = get_simplified_title(train_csv)
train_csv['SimplifiedTitle'].unique()

train_csv['Age'].fillna(train_csv['Age'].median(), inplace=True)
train_csv['AgeBin'] = pd.cut(train_csv['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])
train_csv['Embarked'].fillna(train_csv['Embarked'].mode()[0], inplace = True)
train_csv['Fare'].fillna(train_csv['Fare'].median(), inplace = True)
train_csv['FareBin'] = pd.cut(train_csv['Fare'], bins=[-1, 
                                                  train_csv['Fare'].quantile(.25),
                                                  train_csv['Fare'].quantile(.5), 
                                                  train_csv['Fare'].quantile(.75),
                                                  train_csv['Fare'].max()],
                                                labels=['LowFare', 
                                                        'MediumFare',
                                                        'HighFare', 
                                                        'TopFare'])

train_df = train_csv.copy()
train_df.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'FamilyCount'], axis=1, inplace=True)

#train_df = pd.get_dummies(train_df, columns = ["Pclass", "Sex","Embarked","SimplifiedTitle","AgeBin","FareBin"],
 #                           prefix=["PC", "Sex","Em","ST","Age","Fare"])

def throttling(arr, thres):
    #res = arr.copy()
    res = np.zeros(len(arr))
    res[arr >= thres] = int(1)
    res[arr < thres] = int(0)
    return res

x_train,x_test,y_train,y_test = train_test_split(train_df.drop('Survived', axis=1),
                                                 train_df['Survived'],
                                                 test_size=0.2,
                                                 random_state=123)

#lr = LogisticRegression()
#lr.fit(x_train,y_train)
#y_pred_lr = lr.predict(x_test)
#print('The accuracy of the Logistic Regression is',round(accuracy_score(y_pred_lr,y_test)*100,2))


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

def baselineNN(dims):
    model = Sequential()
    model.add(Dense(10, input_dim=dims, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def use_keras_nn_model(x, y, xx, yy, epochs):
    model = baselineNN(x.shape[1])
    model.fit(x.as_matrix(), y.as_matrix(), epochs=epochs)
    y_pred = model.predict(xx.as_matrix()).reshape(xx.shape[0],)
    return y_pred, model
y_pred_nn, model_nn = use_keras_nn_model(x_train, y_train, x_test, y_test, 100)

print('The accuracy of the Neural Network is',round(accuracy_score(throttling(y_pred_nn, 0.6), y_test)*100,2))