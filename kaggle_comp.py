# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:53:30 2019

@author: ROTIMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn import svm, tree, preprocessing
from sklearn.feature_selection import SelectKBest, chi2, GenericUnivariateSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

train_data = pd.read_csv('promotion_train.csv')

raw_test_data = pd.read_csv('promotion_test.csv')

def show_null_count(csv):
    idx = csv.isnull().sum()
    idx = idx[idx>0]
    idx.sort_values(inplace=True)
    idx.plot.bar()
    
def get_corr(col, csv):
    corr = csv.corr()[col]
    idx_gt0 = corr[corr>0].sort_values(ascending=False).index.tolist()
    return corr[idx_gt0]

def subValYear(selected_column):
    final_list = []
    for x in selected_column:
        if (2019-x) < 25:
            final_list.append(0)
        elif (2019-x) >= 25 and (2019-x) < 45:
            final_list.append(1)
        elif (2019-x) >= 45 and (2019-x) < 75:
            final_list.append(2)
    return pd.Series(final_list)

def subValYear2(selected_column):
    final_list = []
    for x in selected_column:
        if (2019-x) < 5:
            final_list.append(0)
        elif (2019-x) >= 5 and (2019-x) < 10:
            final_list.append(1)
        elif (2019-x) >= 10 and (2019-x) < 15:
            final_list.append(2)
        elif (2019-x) >= 15 and (2019-x) < 20:
            final_list.append(3)
        elif (2019-x) >= 20 and (2019-x) < 25:
            final_list.append(4)
        elif (2019-x) >= 25 and (2019-x) < 30:
            final_list.append(5)
        elif (2019-x) >= 30 and (2019-x) < 35:
            final_list.append(6)
        elif (2019-x) >= 35 and (2019-x) < 40:
            final_list.append(7)
    return pd.Series(final_list)


def subVal(unique_values, selected_column):
    r = {}
    final_list = []
    p = 0
    for ch in unique_values:
        if pd.notna(ch):
            r[ch] = p
            p += 1
    for ch in selected_column:
        if ch in r:
            final_list.append(r[ch])
        else:
            final_list.append(np.nan)
    return pd.Series(final_list)

def pre_process(input_data):
    data = input_data
    division_replace = subVal(data.Division.unique(), data.Division)
    input_data['Qualification'].fillna("First Degree or HND", inplace = True)
    qualification_replace = subVal(data.Qualification.unique(), data.Qualification)
    gender_replace = subVal(data.Gender.unique(), data.Gender)
    channel_replace = subVal(data.Channel_of_Recruitment.unique(), data.Channel_of_Recruitment)
    year_of_birth_replace = subValYear(data.Year_of_birth)
    year_recruit_replace = subValYear(data.Year_of_recruitment)
    state_origin_replace = subVal(data.State_Of_Origin.unique(), data.State_Of_Origin)
    #tsa_replace = subValYear3(input_data['Training_score_average'])
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
    #data['Training_score_average'] = data['Training_score_average']
    data['Foreign_schooled'] = foreign_schooled_replace
    data['State_Of_Origin'] = state_origin_replace    
    data['Marital_Status'] = marital_status_replace
    data['Past_Disciplinary_Action'] = disciplinary_replace
    data['Previous_IntraDepartmental_Movement'] = interdepartmental_replace
    data['No_of_previous_employers'] = previous_emp_replace
    #numeric_features = data.select_dtypes(include=[np.number])
    #numeric_features = numeric_features.interpolate().dropna()
    return data

train_data = pre_process(train_data)

df_majority = train_data[train_data['Promoted_or_Not']==0]
df_minority = train_data[train_data['Promoted_or_Not']==1]
#df_minority_upsampled = resample(df_minority, 
#                                 replace=True,     # sample with replacement
#                                 n_samples=50000,    # to match majority class
#                                 random_state=123)
#train_data = pd.concat([df_majority, df_minority_upsampled])
#
#df_majority_upsampled = resample(df_majority, 
#                                 replace=True,     # sample with replacement
#                                 n_samples=50000,    # to match majority class
#                                 random_state=123)
#train_data = pd.concat([df_majority, df_minority_upsampled])

#df_majority_downsampled = resample(df_majority, 
#                                 replace=False,    # sample without replacement
#                                 n_samples=len(df_minority),     # to match minority class
#                                 random_state=123) # reproducible results
# 
## Combine minority class with downsampled majority class
#train_data = pd.concat([df_majority_upsampled, df_minority_upsampled])



Y = train_data['Promoted_or_Not']
X = train_data.drop(['Promoted_or_Not', 'EmployeeNo'], axis=1)
X = X.drop(['Qualification', 'Trainings_Attended', 'Year_of_recruitment', 'Gender', 'State_Of_Origin'], axis=1)
#X = X.drop(['Year_of_birth', 'Marital_Status', 'No_of_previous_employers', 'Foreign_schooled',
#            'Division', 'Past_Disciplinary_Action', 'Previous_IntraDepartmental_Movement',
#            'Channel_of_Recruitment'], axis=1)



tl = TomekLinks(return_indices=True, ratio='majority')
X_tl, Y_tl, id_tl = tl.fit_sample(X, Y)



#X_tl = pd.DataFrame(X_tl)
#Y_tl = pd.DataFrame(Y_tl)


#cc = ClusterCentroids(ratio={0: 10000})
#X_cc, Y_cc = cc.fit_sample(X, Y)




#smote = SMOTE(ratio='minority')
#X_sm, Y_sm = smote.fit_sample(X, Y)

#smt = SMOTETomek(ratio='auto')
#X_smt, Y_smt = smt.fit_sample(X, Y)


#data_dmatrix = xgb.DMatrix(data=X,label=Y)
    
def dummy(x):
    x = pd.get_dummies(x)
    return x

#X_tl = dummy(X_tl)

X_train, X_test, y_train, y_test = train_test_split(X_tl, Y_tl, test_size=0.01 , random_state=42)



#clf = svm.SVC(gamma='auto')
#clf = LogisticRegression()
#clf = KNeighborsClassifier(n_neighbors=1)



clf=xgb.XGBClassifier(random_state=1,learning_rate=0.1, max_depth=6)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

test_data = pre_process(raw_test_data)
test_data = test_data.drop(['EmployeeNo'], axis=1)
test_data = test_data.drop(['Qualification', 'Trainings_Attended', 'Year_of_recruitment', 'Gender', 'State_Of_Origin'], axis=1)
test_data = test_data.rename(columns={'Division':'f0', 'Channel_of_Recruitment':'f1', 'Year_of_birth':'f2',
       'Last_performance_score':'f3', 'Targets_met':'f4', 'Previous_Award':'f5',
       'Training_score_average':'f6', 'Foreign_schooled':'f7', 'Marital_Status':'f8',
       'Past_Disciplinary_Action':'f9', 'Previous_IntraDepartmental_Movement':'f10',
       'No_of_previous_employers':'f11'})
   
y_test_pred = clf.predict(test_data)
append_list = [['EmployeeNo', 'Promoted_or_Not']]

for x,y in zip(list(raw_test_data.EmployeeNo), list(y_test_pred)):
    append_list.append([x, y])

    
pd.DataFrame(append_list).to_csv('intercampus_submission.csv', encoding='utf-8', index=False)
