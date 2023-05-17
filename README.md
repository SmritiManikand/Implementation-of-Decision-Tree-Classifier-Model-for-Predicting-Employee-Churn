# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## Aim:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Read the data set.
3. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
4. Determine training and test data set.
5. Apply decision tree Classifier and get the values of accuracy and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Smriti M
RegisterNumber: 212221040157
*/

import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

## data.head()

![s1](https://github.com/SmritiManikand/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113674204/079a8329-e536-41b3-9ff3-4ac1daf7f841)

## data.info()

![s2](https://github.com/SmritiManikand/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113674204/a6da82e4-8ec6-493c-a38f-6088d2e072ff)

## isnull() and sum()

![s3](https://github.com/SmritiManikand/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113674204/2241fc4c-bce7-4fa1-bee3-d51c25b87150)

## data value counts()

![s4](https://github.com/SmritiManikand/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113674204/c849a8d4-9e20-4537-aa25-15441e422d8c)

## data.head() for salary

![s5](https://github.com/SmritiManikand/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113674204/2d65ce97-8235-425b-a28e-d46516ff2d73)

## x.head()

![s6](https://github.com/SmritiManikand/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113674204/94e6ba95-c88e-4c7e-9bff-2645d85b9ac4)

## Accuracy value

![s7](https://github.com/SmritiManikand/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113674204/6d5db6d7-d9ab-4384-ad5f-d9127a1c05ff)

## Data prediction

![s8](https://github.com/SmritiManikand/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113674204/eb76a6e1-8003-49b5-b317-2561e57ad8ab)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
