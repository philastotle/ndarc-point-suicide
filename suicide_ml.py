#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:48:50 2019

@author: philliphungerford
"""
# =============================================================================
#  Import dependencies
# =============================================================================
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
import warnings; warnings.simplefilter('ignore') #prevent warnings

# =============================================================================
# Load data
# =============================================================================
def load_data():
    X = pd.read_csv("data/core.csv")
    y = X[['suicide_attempt_12m']]
    y = y.fillna('No')
    X = X.drop(['suicide_attempt_12m'], axis=1)
    return X, y

X,y = load_data()

# Convert categorical to numeric 
X = pd.get_dummies(X)
# Need to split data by id. 
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0, test_size = 0.2)

# =============================================================================
#  Build Tree Model
# =============================================================================
param_grid = {'max_depth': [1,2,3,4,5,6,7,8,9,10],
              'class_weight': [{0:.6, 1:.4}, {0:.62, 1:.38}, {0:.63, 1:.37}, {0:.70, 1:.30}, {0:.73, 1:.27}, ]}  

#1. Use GridSearchCV to find the best 'max_depth' and 'class_weight'.
clf = RandomForestClassifier(n_estimators = 100, max_features = 'sqrt', random_state=0)
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring = 'f1_macro')
grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation average f1 score: {:.2f}".format(grid_search.best_score_))

#2. Train a new classifier using that 'max_depth' and 'class_weight'.
clf = RandomForestClassifier(n_estimators = 100, max_features = 'sqrt', max_depth=2, \
                             class_weight={0:0.63, 1:0.37}, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#3. Assess the classifier in the test set: accuracy, f1 score, f1_macro, precision, recall, and AUC/ROC.
from sklearn import metrics

#Accuracy
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

#alternatively
#print("Accuracy: ", clf.score(X_test, y_test)) 

#f1 score
print("f1 score: ", metrics.f1_score(y_true=y_test, y_pred=y_pred))
#f1_macro
print("f1 macro: ", metrics.f1_score(y_true=y_test, y_pred=y_pred, average="macro"))
#precision
print("Precision: ", metrics.precision_score(y_true=y_test, y_pred=y_pred))
#recall
print("Recall: ", metrics.recall_score(y_true=y_test, y_pred=y_pred))
#AUC/ROC
y_pred_proba = clf.predict_proba(X_test)[:,1]
print("AUC: ", metrics.roc_auc_score(y_true=y_test, y_score=y_pred_proba))
#classification report
print("\n", metrics.classification_report(y_true=y_test, y_pred=y_pred))

#Checking the outcomes
metrics.confusion_matrix(y_test, y_pred)

#4.Display feature importance
print(clf.feature_importances_)
import matplotlib.pyplot as plt

def plot_feature_importances(model):
    
    #locate indices of the features with non-zero feature importance
    indices = np.where(model.feature_importances_ >= 0.02)[0]
    
    #extract the number of features that have non-zero feature importance
    n_features = X.iloc[:,indices].shape[1]
    
    #plot the features that have a non-zero feature importance
    plt.barh(range(n_features), model.feature_importances_[indices], align='center') 
    plt.yticks(np.arange(n_features), X.iloc[:,indices].columns) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    
#plot the feature importance
plot_feature_importances(clf)