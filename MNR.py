import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_excel("C:/Users/sabarishmanogaran/OneDrive - revature.com/Desktop/DS/Project/Procurement Fraudness.xlsx")

#df['Payment terms'].unique()

#df['Payment terms'].value_counts()

df['Unit Price'].unique()

df['Unit Price'].value_counts()

df['InflatedInvoice'].value_counts()

df['InflatedInvoice'].unique()

df['Employees colluding with suppliers with higher cost'].value_counts()

df['Employees colluding with suppliers with higher cost'].unique()

#df['Conflict of Interest'].unique()

#df['Conflict of Interest'].value_counts()

df = df [['Unit Price', 'InflatedInvoice', 'Employees colluding with suppliers with higher cost', 'Fraudness']] 

from sklearn.preprocessing import LabelEncoder

# Creating instance of labelencoder
labelencoder = LabelEncoder()

#df["Payment terms"] = labelencoder.fit_transform(df["Payment terms"])

#df["Conflict of Interest"] = labelencoder.fit_transform(df["Conflict of Interest"])

df.dtypes

df.InflatedInvoice = df.InflatedInvoice.astype('int64')
df.dtypes

df.to_csv('procurementfraudness.csv',encoding="utf-8")
import os
os.getcwd()

train, test = train_test_split(df, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, :3], train.iloc[:, 3])
#help(LogisticRegression)

test_predict = model.predict(test.iloc[:, :3]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:, 3], test_predict)

train_predict = model.predict(train.iloc[:, :3]) # Train predictions 
# Train accuracy 

accuracy_score(train.iloc[:, 3], train_predict)

X = df.iloc[:, :3]

y = df.iloc[:, 3]

regressor = LogisticRegression(multi_class = "multinomial", solver = "newton-cg")

#Fitting model with trainig data
regressor.fit(X, y)


import pickle

# Saving model to disk
pickle.dump(regressor, open('model1.pkl','wb'))

# Loading model to compare the results
model1 = pickle.load(open('model1.pkl','rb'))
print(model1.predict([[170,489,1849]]))
