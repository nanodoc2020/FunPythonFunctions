# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 21:42:54 2020
Seeds is a k-Nearest Neighbors model that attempts to classify a type of 
seed as either Kama, Rosa, or Canadian wheat by using seven geometric 
properties of the seeds. They are, according to their columns, 1)area 
2)perimeter 3)compactness (4*pi*A/p^2) 4)length 5)width 6)asymmetry 
coefficient 7)groove length 8)class. The data set used is "seeds" from the
UCI Machine Learning Repository. 
@author: Erik Larsen
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Create a list for the class names:
classes=['Kama','Rosa','Canadian']

#Import the training set data and create a DataFrame:
seeds=pd.read_csv('Seeds_data.csv')

#Assign the proper columns:
X=seeds.drop('class', axis=1).values
y=seeds['class'].values

#Split the data into training and test sets:
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,\
                                                   random_state=11,stratify=y)

#A quick histogram to visualize the data (created using Orange):
plt.figure(figsize=(8,6))
plt.imshow(img.imread('seeds_hist.png'))
plt.title('Comparing Length & Groove Length')
plt.show()

#Choose the best value for number of nearest neighbors:
"""The best score is for when n_neighbors=5 or 7. """
scr_lst=[]
for n in range(3,15):
    knn=KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train,y_train)
    scr_lst.append(knn.score(X_test,y_test))

plt.figure()
plt.scatter(np.arange(3,15),scr_lst)
plt.title('Scores w/ Different k Values')
plt.xlabel('# of nearest neighbors (k)')
plt.ylabel('accuracy score')
plt.show()

#Create the classifier and fit to the training data:
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

#Use the trained model to predict on unseen data:
prediction = list(knn.predict(X_test))

#Populate a list of names for the predictions:
pred_lst=[classes[i-1] for i in prediction]

#Analyse the results:
score = round(knn.score(X_test,y_test),3)*100
comparison=prediction==y_test
results=list(zip(pred_lst,comparison))

#Print the results and analysis:
print('\nResults using a kNN approach:')
print(results,'\n\n Score:',score,'%')

















