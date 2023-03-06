# -*- coding: utf-8 -*-
"""
Implementation of the majority vote classfier in Scikit-learn 
Author: Aleksandar Haber

Note that this code file imports the function "visualizeClassificationAreas"
from the file: functions.py


"""
# support vector machine classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import SVC
# logistic regression classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
# random forest classifer
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
# voting classifier 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
from sklearn.ensemble import VotingClassifier
# data set database for generating different data sets for testing the algorithms
# https://scikit-learn.org/stable/datasets.html
from sklearn import datasets
# accuracy_score metric to test the performance of the classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
from sklearn.metrics import accuracy_score
# train_test split
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split
# standard scaler used to scale and standardize the data set
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler 

# function for visualizing the classification areas
from functions import visualizeClassificationAreas

# load the data set
# there are two data sets that we tested

# 1. Iris data set 
# https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
# 2. Moons data set
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html

# Iris data set
#dataSet=datasets.load_iris()
# input data for classification
#Xtotal=dataSet['data'][:, 1:3]
#Ytotal=dataSet['target']

# Moons data set
Xtotal, Ytotal = datasets.make_moons(n_samples=200, noise = 0.15)


# split the data set into training and test data sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtotal, Ytotal, test_size=0.5)

# create a standard scaler
scaler1=StandardScaler()
# scale the training and test input data
# fit_transform performs both fit and transform at the same time
XtrainScaled=scaler1.fit_transform(Xtrain)
# here we only need to transform
XtestScaled=scaler1.transform(Xtest)

# create the classifiers
# support vector machines
SVMCLF=SVC(decision_function_shape='ovo')
# logistic regression
LogisticCLF=LogisticRegression(random_state=42)
# random forrest classifier
ForestCLF=RandomForestClassifier(n_estimators=30)

# create a list of classifier tuples
# this list is used to form the voting classifier
# (classifier name, classifier object)
classifierTypeNameInitial=[('SVM',SVMCLF),('LogisticRegression',LogisticCLF),('RandomForest',ForestCLF)]
# here we create the voting classifier
VotingCLF=VotingClassifier(estimators=classifierTypeNameInitial, voting='hard')

# create the final list of tuples of classifiers
classifierTypeNameTotal=classifierTypeNameInitial+[('Voting',VotingCLF)]

# this dictionary is used to store the classification scores
classifierScore={}

# here we iterate through the classifiers and compute the accuracy score
# and store the accuracy store in the list
for nameCLF,CLF in classifierTypeNameTotal:
    CLF.fit(XtrainScaled,Ytrain)
    CLF_prediction=CLF.predict(XtestScaled)
    classifierScore[nameCLF]=accuracy_score(Ytest,CLF_prediction)

# visualize the classification regions

visualizeClassificationAreas(SVMCLF,XtrainScaled, Ytrain,XtestScaled, Ytest, filename='classification_results_svm.png', plotDensity=0.01 )
visualizeClassificationAreas(LogisticCLF,XtrainScaled, Ytrain,XtestScaled, Ytest, filename='classification_results_logistic.png', plotDensity=0.01 )
visualizeClassificationAreas(ForestCLF,XtrainScaled, Ytrain,XtestScaled, Ytest, filename='classification_results_forest.png', plotDensity=0.01 )
visualizeClassificationAreas(VotingCLF,XtrainScaled, Ytrain,XtestScaled, Ytest, filename='classification_results_voting.png', plotDensity=0.01 )


