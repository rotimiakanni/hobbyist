## -*- coding: utf-8 -*-
#"""
#Created on Fri May 17 14:41:51 2019
#
#@author: ROTIMI
#"""
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn import svm
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from sklearn import preprocessing
#
#def visual_data(dataset):
#    sepal_l = np.array(dataset["sepal.length"])
#    sepal_w = np.array(dataset["sepal.width"])
#    petal_l = np.array(dataset["petal.length"])
#    petal_w = np.array(dataset["petal.width"])
#    label = np.array(dataset["variety"])
#    
#    plt.plot(sepal_l[label=='Setosa'], sepal_w[label=='Setosa'], 'bo')
#    plt.plot(sepal_l[label=='Versicolor'], sepal_w[label=='Versicolor'], 'ro')
#    plt.plot(sepal_l[label=='Virginica'], sepal_w[label=='Virginica'], 'yo')
#    plt.xlabel("sepal length")
#    plt.ylabel("sepal width")
#    
#if __name__ == '__main__':
#    dataset = pd.read_csv("iris.csv")
#    X = np.array(dataset[["sepal.length", "sepal.width", "petal.length", "petal.width"]])
#    y = np.array(dataset[["variety"]])
#    
#    clf = svm.SVC(gamma='scale')
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#    clf.fit(X_train, y_train)
#    y_pred = clf.predict(X_test)
##    plt.plot(X_train["sepal.length"], X_train["sepal.width"])
##    plt.show()
#    print(accuracy_score(y_test, y_pred))
#    
#    
#    ##### normLIZATION OF DATA
##    for n in ['Standardization', 'MinMax', 'Quantile']:
##        if n == 'Standardization':
##            normalizer = preprocessing.StandardScaler()
##        elif n == 'MinMax':
##            normalizer = preprocessing.MinMaxScaler()
##        else:
##            normalizer = preprocessing.QuantileTransformer()
##        X_train_norm = normalizer.fit_transform(X_train)
##        X_test_norm = normalizer.transform(X_test)
##        clf_rbf = svm.SVC(kernel='rbf', gamma='scale').fit(X_train_norm, y_train)
##        print(accuracy_score(y_test, clf_rbf.predict(X_test_norm)))
#    
#    
h = {'a':['appl', 'adf', 'hagf'], 'b':['mhdf', 'jgdv'] 
def conv(dta):
    for i in dta:
        for x in i:
            if x

    
    
    
