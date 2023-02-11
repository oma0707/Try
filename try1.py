import pandas as pd
import sklearn
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

import numpy as np
np.random.seed(42)

from sklearn.model_selection import train_test_split
path_to_file="C:/CMI/SEM 6/DMML/Assignment1/bank-data/bank-additional-full.csv"
bank_data=pd.read_csv(path_to_file, sep=";")
target_names = bank_data.iloc[0:5,0:5]


training_data, testing_data = train_test_split(bank_data, random_state=25)
training_data=training_data.drop(columns='duration')
training_data=training_data.drop(columns='month' )
training_data=training_data.drop(columns='day_of_week')
training_data=training_data.drop(columns='contact')
training_data=training_data.drop(columns='campaign')

#Changing categorical attributes to numerical attributes using Label Encoding.
categorical_cols = ['job', 'marital', 'education', 'default', 'poutcome', 'housing', 'default', 'loan']
training_data = pd.get_dummies(training_data, columns = categorical_cols, drop_first = True)

x = training_data.drop(["y"], axis = 1)
y = training_data["y"]

feature_names = x.columns
print(training_data)
print(feature_names)

# Importing DecisionTree module from Scikit-Learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
x_train=training_data.drop(["y"],axis=1)
print(x_train)
y_train=training_data["y"]
acc_list=[]
rec_list=[]
prec_list=[]
testing_data=testing_data.drop(columns='duration')
testing_data=testing_data.drop(columns='month' )
testing_data=testing_data.drop(columns='day_of_week')
testing_data=testing_data.drop(columns='contact')
testing_data=testing_data.drop(columns='campaign')

#Changing categorical attributes to numerical attributes using Label Encoding.
categorical_cols = ['job', 'marital', 'education', 'default', 'poutcome', 'housing', 'default', 'loan']
testing_data = pd.get_dummies(testing_data, columns = categorical_cols, drop_first = True)
x_test=testing_data.drop(["y"],axis=1)
y_test=testing_data["y"]
n_cols=len(testing_data.columns)

DT = DecisionTreeClassifier(max_depth = 3, max_leaf_nodes = 16, random_state=42)
DT.fit(x_train, y_train)
DecisionTreeClassifier(max_depth=3, max_leaf_nodes=16, random_state=42)
y_pred = DT.predict(x_test)
y_test = np.array(y_test).flatten()

print(classification_report(y_test, y_pred))

print(max(acc_list))
print(max(prec_list))