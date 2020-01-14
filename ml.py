# -*- coding: utf-8 -*-
"""
Created on Fri May 17 02:45:35 2019

@author: ROTIMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#def visualize(data):

#if __name__ == "__main__":
#    dataset = pd.read_csv("iris.csv")
#    sepal_l = np.array(dataset['sepal.length'])
#    sepal_w = np.array(dataset['sepal.width'])
#    petal_l = np.array(dataset['sepal.length'])
#    petal_w = np.array(dataset['sepal.width'])
#    labels = np.array(dataset['variety'])
#    
#    plt.plot(sepal_l[labels=='Setosa'], sepal_w[labels=='Setosa'], 'ro')
#    plt.plot(sepal_l[labels=='Virginica'], sepal_w[labels=='Virginica'], 'bo')
#    plt.plot(sepal_l[labels=='Versicolor'], sepal_w[labels=='Versicolor'], 'go')
#    plt.xlabel('sepal length')
#    plt.ylabel('sepal length')
#    plt.title('iris plot')
##    plt.plot()
#    
#    X = np.array(dataset[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']])
#    Y = np.array(dataset['variety'])
#    clf = svm.SVC(gamma='scale')
#    
#    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
#    
#    clf.fit(X_test, Y_test)
#    
#    Y_pred = clf.predict(X_test)
#    
#    acc = accuracy_score(Y_test, Y_pred)
#    
#    print(acc)

import os
import shutil

#%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# List all files in a directory using os.listdir
basepath = 'C:\\Users\\ROTIMI\\Desktop\\AIC\\train\\5\\'
#for entry in os.listdir(basepath):
#    if os.path.isfile(os.path.join(basepath, entry)):
#        print(entry)
        
        

datas = pd.read_csv("test.csv")        
#for x in range(5):
#    img=mpimg.imread(datas.image[x])
#    imgplot = plt.imshow(img)  
#    plt.show()
    
    
dir1 = 'C:\\Users\\ROTIMI\\Desktop\\AIC\\train\\5val\\'
#count = 0
#for x,y in zip(datas.image, datas.category):
#    if x in os.listdir(basepath) and y == 1:
#        count += 1
##print(count)
#        shutil.move(os.path.join(basepath, x), dir1) 
#
#print(count)

u = os.listdir(basepath)
#print(0.3*len(u))
g = u[0:int((0.3*len(u)))]

for x in g:
    shutil.move(os.path.join(basepath, x), dir1)
        
print(len(u))    
print(len(g))
    
    
    
    
    
    
    
    
    
    
    
    
    

        