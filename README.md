# Cardiovascular Disease Prediction Using Machine Learning

## Project Overview

The "Cardiovascular Disease Prediction Using Machine Learning" project aims to develop a predictive model that can determine whether a patient is at risk of cardiovascular diseases (CVD). By utilizing various machine learning algorithms, the project helps healthcare professionals assess the likelihood of heart disease based on key health metrics such as age, cholesterol levels, blood pressure, and lifestyle factors. This project is an important tool for early diagnosis and intervention, which can significantly reduce the mortality rate caused by heart diseases.

## Features
- **Multiple Algorithms**: The project implements multiple machine learning algorithms to compare their performance, including Decision Tree, Random Forest, Linear Regression, and Logistic Regression.
- **Predictive Analysis**: Predicts whether a patient has cardiovascular disease based on user input (such as age, cholesterol levels, and blood pressure).
- **Data Preprocessing**: Handles missing data and splits the dataset into training and testing sets for model validation.
- **User-friendly Interface**: Allows input from the user to make predictions using the trained machine learning models.
- **Model Comparison**: Evaluates the accuracy of different models, helping determine the best one for heart disease prediction.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Modeling Process](#modeling-process)
- [Technologies Used](#technologies-used)
- [Results](#results)

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aruljothi156/Cardiovascular_diesease_prediction.git
2.**Install dependencies: Install the required libraries using pip:**
   bash
  pip install numpy pandas scikit-learn
3.Download the dataset:
  .Ensure you have the Cardio.csv dataset file in the project directory.

## Data

The dataset `Cardio.csv` contains several health-related attributes of patients that are correlated with cardiovascular diseases. Some of the key features include:

- **Age**: Age of the patient (continuous variable).
- **Sex**: Gender of the patient (0 = female, 1 = male).
- **Chest Pain Type**: Type of chest pain experienced by the patient (categorical: 1, 2, 3, 4).
- **Resting Blood Pressure**: Blood pressure measured while the patient is at rest (continuous variable).
- **Serum Cholesterol**: Level of serum cholesterol in mg/dl (continuous variable).
- **Fasting Blood Sugar**: Blood sugar level after fasting (1 if greater than 120 mg/dl, 0 otherwise).
- **Resting Electrocardiographic Results**: Results of the resting electrocardiogram (ECG) (categorical: 0, 1, 2).
- **Maximum Heart Rate Achieved**: The highest heart rate recorded during stress testing (continuous variable).
- **Exercise Induced Angina**: Whether exercise induced angina was present (1 = yes, 0 = no).
- **ST Depression Induced by Exercise Relative to Rest**: ST depression observed during exercise testing (continuous variable).
- **Slope of the Peak Exercise ST Segment**: Type of slope for the peak exercise ST segment (categorical: 1, 2, 3).
- **Number of Major Vessels Colored by Fluoroscopy**: Number of major vessels colored by fluoroscopy (categorical: 0, 1, 2, 3).
- **Thalassemia**: Whether the patient has thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect).
- **Target**: Whether the patient has cardiovascular disease (1 = disease present, 0 = no disease).

## Usage

### Import Required Libraries

To run the project, you'll need to import the following libraries:

python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

#Load the Data

###The dataset, Cardio.csv, contains the necessary information about patients' health indicators.

python
heartdata = pd.read_csv("heart.csv")
heartdata.head()  # View the first few rows of the dataset
heartdata.tail()  # View the last few rows of the dataset

#Data Exploration

Perform basic data exploration to understand the structure of the dataset:

 python
 heartdata.info()  # Get information about the dataset
 heartdata.describe()  # Get descriptive statistics of the dataset

 #Data Preparation

 ###Separate the features (X) and target variable (Y):

 python
 X = heartdata.drop(columns='target', axis=1)  # Features (all columns except 'target')
 Y = heartdata['target']  # Target column (whether the patient has heart disease or not)

 Split the data into training and testing sets:

 python
 X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

 #Modeling Process

 ###Decision Tree Classifier

Train a Decision Tree model and evaluate its accuracy:

python
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(Y_test, y_pred_dt)
print("Decision Tree accuracy:", acc_dt)

#Random Forest Classifier

###Train a Random Forest model and evaluate its accuracy:

 python
 rf = RandomForestClassifier()
 rf.fit(X_train, Y_train)
 y_pred_rf = rf.predict(X_test)
 acc_rf = accuracy_score(Y_test, y_pred_rf)
 print("Random Forest accuracy:", acc_rf)

 #Linear Regression (For Binary Classification)

###Although linear regression is typically used for regression tasks, here we use it for binary classification:

python
lr = LinearRegression()
lr.fit(X_train, Y_train)
y_pred_lr = lr.predict(X_test)
y_pred_lr[y_pred_lr < 0.5] = 0  # Convert continuous values to 0 or 1
y_pred_lr[y_pred_lr >= 0.5] = 1
acc_lr = accuracy_score(Y_test, y_pred_lr)
print("Linear Regression accuracy:", acc_lr)

#Logistic Regression

###Train a Logistic Regression model and evaluate its accuracy:

python
model = LogisticRegression()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
train_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Test Data Accuracy:", test_accuracy)

#Prediction for a New User

###You can also use the trained model to predict the likelihood of heart disease for a new patient based on their health data:

python
input_from_user = (43, 0, 0, 132, 341, 1, 0, 136, 1, 3, 1, 0, 3)
input_from_user_array = np.asarray(input_from_user)
input_from_user_reshaped = input_from_user_array.reshape(1, -1)
prediction = model.predict(input_from_user_reshaped)

if prediction[0] == 0:
    print("Patient does not have any heart disease.")
else:
    print("Patient has heart disease, further tests are recommended.")
    

#Technologies Used

- **Python**:Programming language used for data processing and machine learning.
- **Pandas**: Used for data manipulation and analysis.
- **NumPy**:Used for numerical operations on data.
- **Scikit-learn**:Used for machine learning algorithms and model evaluation.

#Results

###The project uses multiple machine learning algorithms, and their performance is evaluated as follows:

- **Decision Tree Accuracy**:1.0
- **Random Forest Accuracy**:1.0
- **Linear Regression Accuracy**:0.8048
- **Logistic Regression Accuracy**:0.8048










 







 




