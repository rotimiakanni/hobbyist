# -*- coding: utf-8 -*-
"""
Created on Tue May 21 01:12:55 2019

@author: ROTIMI
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def save_csv(filename, x, y):
    f = open(filename, 'w')
    for ele_x, ele_y in zip(x,y):
        f.writelines([str(ele_x) + ',' + str(ele_y) + '\n' ])
    return

def load_data(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    x, y = [], []
    for line in lines:
        first_e, second_e = line.strip().split(',')
        x.append(float(first_e))
        y.append(float(second_e))
    return x, y
    
    
if __name__ == '__main__':
#    n = 20
#    x = np.random.uniform(0, 20, size=[n])
#    y = 2*x + np.random.normal(loc=0.0, scale=5, size=[n])
#    save_csv('data.csv', x, y)
    x, y = load_data('data.csv')
    x = np.array(x).reshape([-1,1])
    y = np.array(y).reshape([-1,1])
    #print(x)
    reg = LinearRegression().fit(x, y)
    print(reg)
    x_new = np.random.uniform(-5, 30, size=[50]).reshape([-1,1])
    y_new = reg.predict(x_new)
    
    plt.plot(x, y, 'rx')
    plt.plot(x_new, y_new)
    
    