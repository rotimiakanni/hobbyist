# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:49:16 2019

@author: ROTIMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
from sklearn import svm, tree, preprocessing
from sklearn.feature_selection import SelectKBest, chi2, GenericUnivariateSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

train_data = pd.read_csv('promotion_train.csv')
train_data = train_data.drop(['EmployeeNo'], axis=1)
raw_test_data = pd.read_csv('promotion_test.csv')

def subVal(unique_values, selected_column):
    r = {}
    final_list = []
    p = 1
    for ch in unique_values:
        if pd.notna(ch):
            r[ch] = p
            p += 1
    for ch in selected_column:
        if ch in r:
            final_list.append(r[ch])
        else:
            final_list.append(np.nan)
    return final_list
    
def subValYear(selected_column):
    final_list = []
    for x in selected_column:
        final_list.append(2019-x)
    return final_list

def pre_process(input_data):
    data = input_data
    division_replace = subVal(data.Division.unique(), data.Division)
    qualification_replace = subVal(data.Qualification.unique(), data.Qualification)
    gender_replace = subVal(data.Gender.unique(), data.Gender)
    channel_replace = subVal(data.Channel_of_Recruitment.unique(), data.Channel_of_Recruitment)
    year_of_birth_replace = subValYear(data.Year_of_birth)
    year_recruit_replace = subValYear(data.Year_of_recruitment)
    state_origin_replace = subVal(data.State_Of_Origin.unique(), data.State_Of_Origin)
    foreign_schooled_replace = subVal(data.Foreign_schooled.unique(), data.Foreign_schooled)
    marital_status_replace = subVal(data.Marital_Status.unique(), data.Marital_Status)
    disciplinary_replace = subVal(data.Past_Disciplinary_Action.unique(), data.Past_Disciplinary_Action)
    interdepartmental_replace = subVal(data.Past_Disciplinary_Action.unique(), data.Past_Disciplinary_Action)
    previous_emp_replace = subVal(data.No_of_previous_employers.unique(), data.No_of_previous_employers)
    
    data['Division'] = division_replace
    data['Qualification'] = qualification_replace
    data['Gender'] = gender_replace
    data['Channel_of_Recruitment'] = channel_replace
    data['Year_of_birth'] = year_of_birth_replace
    data['Year_of_recruitment'] = year_recruit_replace
    data['Foreign_schooled'] = foreign_schooled_replace
    data['State_Of_Origin'] = state_origin_replace    
    data['Marital_Status'] = marital_status_replace
    data['Past_Disciplinary_Action'] = disciplinary_replace
    data['Previous_IntraDepartmental_Movement'] = interdepartmental_replace
    data['No_of_previous_employers'] = previous_emp_replace
#    data = data.drop(['Qualification'], axis=1)
    numeric_features = data.select_dtypes(include=[np.number])
#    numeric_features = normalize(numeric_features)
    numeric_features = numeric_features.interpolate().dropna()
    
#    numeric_features = numeric_features[numeric_features['Training_score_average']<90]
    
#    
#    numeric_features = numeric_features.dropna()
    return numeric_features

def outliers(numeric_features):
#    numeric_features['Year_of_birth'] = np.log(numeric_features['Year_of_birth'])
#    numeric_features['Year_of_recruitment'] = np.log(numeric_features['Year_of_recruitment'])
#    numeric_features = numeric_features[numeric_features['Trainings_Attended']<5]
    return numeric_features
    
train_data = pre_process(train_data)
train_data = outliers(train_data)

Y = train_data['Promoted_or_Not']
X_old = train_data.drop(['Promoted_or_Not'], axis=1)

print


scaler = MinMaxScaler()
#X_old = X_old.apply(lambda x: scaler.fit_transform(x))

def scaleColumns(df, cols_to_scale):
    for col in cols_to_scale:
        df[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(X_old[col])),columns=[col])
    return df

X_old = scaleColumns(X_old, X_old.columns)

#X_old = X_old.apply(lambda x: scaler.transform(x))

#pca = PCA()
#pca.fit(data_rescaled)
#print(pca.explained_variance_ratio_)


#selector = SelectKBest(GenericUnivariateSelect, k=2)

#selector = GenericUnivariateSelect(chi2, 'k_best', param=8)
#X = selector.fit_transform(X_old, Y)

#X = (selector.fit_transform(X_old, Y))



#mask = selector.get_support()
#new_features = []
#
#for bool, feature in zip(mask, X_old.columns):
#    if bool:
#        new_features.append(feature)

X_train, X_test, y_train, y_test = train_test_split(X_old, Y, test_size=0.001, random_state=42)

#clf = LogisticRegression(solver='newton-cg', max_iter=200, random_state=40)
clf = svm.SVC(gamma='scale')
#clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

test_data = pre_process(raw_test_data)
test_data = scaleColumns(test_data, test_data.columns)
#test_data = test_data[new_features]
y_test_pred = clf.predict(test_data)

append_list = [['EmployeeNo', 'Promoted_or_Not']]

for x,y in zip(list(raw_test_data.EmployeeNo), list(y_test_pred)):
    append_list.append([x, y])

    
pd.DataFrame(append_list).to_csv('intercampus_submission.csv', index=False)



#column_list = []
#other_list = []
#counter = 0
#new_feature = pd.DataFrame(X).astype(int)
#old_feature = X_old
#old_col = old_feature.columns
#for x in old_col:
#    for y in range(len(new_feature.columns)):
#        if list(old_feature[x])[:10] == list(new_feature[y])[:10]:
#            column_list.append(x)
#            other_list.append(y)
#        counter += 1
#
#
#mask = X.get_support() #list of booleans



