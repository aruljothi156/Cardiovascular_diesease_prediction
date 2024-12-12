import numpy as np
import pandas as py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
heartdata=py.read_csv("Cardio.csv")
heartdata.head()
heartdata.tail()
# heartdata.shape
heartdata.info()
heartdata.describe()
targets=heartdata['target'].value_counts()
#all columns
X=heartdata.drop(columns='target',axis=1)
#target column
Y=heartdata['target']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
# print(X.shape,X_train.shape,X_test.shape)
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(Y_test, y_pred_dt)
print("Decision Tree accuracy:", acc_dt)
# Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(Y_test, y_pred_rf)
print("Random Forest accuracy:", acc_rf)
# Linear Regression model
lr = LinearRegression()
lr.fit(X_train, Y_train)
y_pred_lr = lr.predict(X_test)
y_pred_lr[y_pred_lr < 0.5] = 0
y_pred_lr[y_pred_lr >= 0.5] = 1
acc_lr = accuracy_score(Y_test, y_pred_lr)
print("Linear Regression accuracy:", acc_lr)
# model = LogisticRegression()
# model.fit(X_train, Y_train)
# y_pred_lr = model.predict(X_test)
# acc_lr = accuracy_score(Y_test, y_pred_lr)
# print("Logistic Regression accuracy:", acc_lr)
model=LogisticRegression()
model.fit(X_train,Y_train )
X_train_prediction=model.predict(X_train)
trainigdataaccuracy=accuracy_score(X_train_prediction,Y_train)
# print( trainigdataaccuracy)
X_test_prediction=model.predict(X_test)
testdataaccuracy=accuracy_score(X_test_prediction,Y_test)
print( testdataaccuracy)
input_from_user=(43,0,0,132,341,1,0,136,1,3,1,0,3)
input_from_user_array=np.asarray(input_from_user)
input_from_user_reshaped=input_from_user_array.reshape(1,-1)
prediction=model.predict(input_from_user_reshaped)
if prediction[0]==0:
    print("Patient Doesnot have Any Heart Dieseas")
else:
    print("Patient Has heart dieseas he needs more tests")
