#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#Importing dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Combining test and train datasets to feature engineer them together
dataset = train.drop('Survived', axis=1)
dataset = pd.concat([dataset, test])

#Function to calculate total number of family member per person. SibSp+Parch+1
def calculate_family_members(cols):
    SibSp = cols[0]
    Parch = cols[1]
    return (SibSp+Parch+1)

#Function to calculate fare per person. In case of family the fare column gives combined fare. Fare/(SibSp+Parch+1)
def real_fare(cols):
    SibSp = cols[0]
    Parch = cols[1]
    Fare = cols[2]
    Pclass = cols[3]
    #If Fare is nan then assiged the average fare based on Pclass
    if (math.isnan(Fare)):
        if Pclass == 3:
            Fare = 13.30
        elif Pclass == 2:
            Fare = 21.18
        else:
            Fare = 87.51
    
    return Fare / (SibSp+Parch+1)

#Function to get only the Title(Mr, Mrs etc) from name. 
def split_title(name):
    title = ((name.split(', ')[1]).split('.')[0])
    return title

#Function to decrease the number of titles to reasonable number.
def remove_duplicate_title(cols):
    Title = cols[0]
    Sex = cols[1]
    if Title == 'Mme':
        return 'Mlle'
    if Sex == 'male':
       if Title in ['Don', 'Rev', 'Dr', 'Major', 'Col', 'Capt', 'Jonkheer']:
           return 'Sir'
    else:
        if Title in ['Dr', 'the Countess', 'Jonkheer', 'Dona']:
           return 'Lady'
    return Title

#Function to get age in case of nan with average age  based on class and sex
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    Sex = cols[2]
    if pd.isnull(Age):
        if Pclass == 1 and Sex=='male':
            return 40
        elif Pclass == 1 and Sex == 'female':
            return 35
        elif Pclass == 2 and Sex == 'male':
            return 30
        elif Pclass == 2 and Sex == 'female':
            return 27
        elif Pclass == 3 and Sex == 'male':
            return 25
        else:
            return 22
    else:
        return Age

#Function to get surname from name
def split_surname(name):
    surname = (name.split(', ')[0])
    return surname

#Fucntion to remove nan entries for Embarked
def remove_nan_embark(embarked):
    if embarked not in ['S', 'Q', 'C']:
        return 'S'
    else:
        return embarked

dataset['Title'] = dataset['Name'].apply(split_title)
dataset['Title'] = dataset[['Title','Sex']].apply(remove_duplicate_title, axis=1)
dataset['FMembers']= dataset[['SibSp','Parch']].apply(calculate_family_members,axis=1)
dataset['RFare']= dataset[['SibSp','Parch','Fare','Pclass']].apply(real_fare,axis=1)
dataset['Age']= dataset[['Age', 'Pclass', 'Sex']].apply(impute_age,axis=1)
dataset['Embarked']= dataset['Embarked'].apply(remove_nan_embark)

#Using dummy variables for non numeric data
title = pd.get_dummies(dataset['Title'], drop_first=True)
sex = pd.get_dummies(dataset['Sex'], drop_first=True)
embarked = pd.get_dummies(dataset['Embarked'], drop_first=True)

#Concatenate new columns
dataset = pd.concat([dataset,sex,title, embarked],axis=1)

#Remove unused columns
dataset.drop(['Sex','Embarked','Name','Ticket', 'Fare','Title','Cabin'],axis=1,inplace=True)

#Split train and test data
X_train = dataset.iloc[0:891, :].values
X_test = dataset.iloc[891:, :].values
y_train = train.iloc[:, 1].values


#Variable scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

