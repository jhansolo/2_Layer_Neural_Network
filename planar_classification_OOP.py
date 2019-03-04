# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 07:05:58 2019

an exercise in creating a 2 layer neural network classifier
object following the derivations and implementations for
gradient descent from the Deep Learning course taught by Andrew Ng 
on Coursera. 
Replicates the 'flower'-shaped, planar, binary classification 
challenge. Accuracy up to 96%, depending on initial dataset noise. 
Optimal hidden layer sizeat 5. No additional benefit observed for 
larger size

@author: jh
"""
import NeuralNetwork            #stores the functions and objects necessary

data=NeuralNetwork.Data()       #creates data object
X=data.feature
Y=data.label
plotting=True                       #for visualize, set to False if dim_feature>2

n=0.1                           #portion of data to set aside for testing
maxCycles=1000                  #max iterations, rarely fully reached
minCycles=100                   #min number of iterations, essential for early-stopping
alpha=.5                        #learning rate
tol=1e-5                        #to compare each iteration's cost again'st the previous, used for early stopping

#split into training, testing data
X_train,X_test,y_train,y_test=NeuralNetwork.split(X,Y,n)

n_h=5                           #hidden layer size
n_y=1                           #output layer size, 1 for binary classification

#calling upon classifier object, see NeuralNetwork.py for details/documtation
clf=NeuralNetwork.NN(n_h,n_y,maxCycles,minCycles,alpha,tol)
clf.fit(X_train,y_train)
clf.predict(X_test,y_test)
clf.printOut()

if plotting is True:
    NeuralNetwork.plot(X,data.co,clf.error)





