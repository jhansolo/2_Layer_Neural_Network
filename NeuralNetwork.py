# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 07:58:37 2019

contains the functions and classes necessary for implementing
the 2-layer neural network binary classifier taught by Andrew
Ng in the deep learning course on Coursra

@author: jh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
import copy
import time


class Data():
    """creates data object, a very simplified version of sklearn data"""
    def __init__(self):
        df=pd.read_csv('planar_data.csv')
        self.X=df['X']
        self.Y=df['Y']
        self.Z=df['Z']
        self.feature=np.stack((self.X.values,self.Y.values))    
        self.label=self.Z.values
        self.label=self.label.reshape(1,len(self.label))
        
        """visualization: z=1 ->blue dots, z=0 ->red dots"""
        self.co=[]
        for i in self.Z:
            red='r'
            blue='b'
            if i>0.5:
                self.co.append(red)
            else:
                self.co.append(blue)

    def plot(self):        
        plt.style.use('classic')
        plt.scatter(self.X,self.Y,c=self.co,s=30)
        plt.axis('off')
        plt.axis('equal')

class NN():
    """neural network classifier object"""
    
    def __init__ (self,n_h=5,n_y=1,maxCycles=5000,minCycles=200,alpha=0.5,tol=1e-4):
        self.n_h=n_h
        self.n_y=n_y
        self.maxCycles=maxCycles
        self.minCycles=minCycles
        self.alpha=alpha
        self.tol=tol
    
    def fit (self,X_train,y_train):
        """fitting data"""
        tick=time.process_time()
        self.error=[]                       #accumulates the cost function per iteration
        self.X_train=X_train
        self.n_x=len(self.X_train)
        self.y_train=y_train
        self.m=self.X_train.shape[1]        #total number of training samples
        self.W1,self.B1,self.W2,self.B2=initialize(self.n_x,self.n_h,self.n_y)      #random initializing weight matrix, zeros wouldn't work
        self.Z1,self.A1,self.Z2,self.A2=forwardProp(self.W1,self.B1,self.W2,self.B2,self.X_train) #do one forward prop, just to get things started
        self.cost=cost(self.A2,self.y_train,self.m) #record the cost of this initial step, used for comparsion with the second iteration
        
        for self.i in range(self.maxCycles):
            self.error.append(self.cost)    #accomulates cost
            oldCost=copy.deepcopy(self.cost) #make a copy of previous cost
            #computes gradients of the respective weight matrices and bias vectors 
            self.dW2,self.dB2,self.dW1,self.dB1=backProp(self.m,self.A2,self.W2,self.y_train,self.A1,self.X_train)
            #updates the weight matrices and bias vectors with above info and learning rate
            self.W2,self.B2,self.W1,self.B1=update(self.W2,self.dW2,self.B2,self.dB2,self.W1,self.dW1,self.B1,self.dB1,self.alpha)
            #computes new outputs
            self.Z1,self.A1,self.Z2,self.A2=forwardProp(self.W1,self.B1,self.W2,self.B2,self.X_train)
            #compute new cost
            self.cost=cost(self.A2,self.y_train,self.m)
            #comparsion between new cost and old cost. If difference below tolerance, stop early. Note that 
            #the differences in the first n cycles are very small, thus if there isn't a minCycle, the tolerances
            #would have to be made excessively small or the loop would never start
            if self.i>self.minCycles:
                if abs(self.cost-oldCost)<self.tol:
                    break
        
        #found the optimal weight matrices and bias vectors, now use to do prediction on the training data for performance assesment
        y_train_pred=(self.A2>0.5)
        y_train_pred.astype(int)
        diff=y_train_pred-self.y_train
        self.wrongTrain=np.count_nonzero(diff)
        self.rightTrain=self.m-self.wrongTrain
        self.elapsedTime=time.process_time()-tick
        
    def predict(self,X_test,y_test):
        """found the optimal weight matrices and bias vectors, now predicton the testing data"""
        self.X_test=X_test
        self.y_test=y_test
        self.Z1_p,self.A1_p,self.Z2_p,self.A2_p=forwardProp(self.W1,self.B1,self.W2,self.B2,self.X_test)
        y_test_p=self.A2_p>0.5
        diff=y_test_p-self.y_test
        self.wrongGuess=np.count_nonzero(diff)
        self.rightGuess=self.X_test.shape[1]-self.wrongGuess

    def printOut(self):
        """output"""
        print("=============================")
        print("TRAINING")
        print("=============================")
        print(self.i, " cycles")
        print(self.elapsedTime,'seconds')
        print("number of wrong guesses: ",self.wrongTrain)
        print("number of right guesses: ",self.rightTrain)
        print("accuracy: {}%".format(np.round(self.rightTrain*100/(self.wrongTrain+self.rightTrain),2)))
        
        print("=============================")
        print("Testing")
        print("=============================")
        print("number of wrong guesses: ",self.wrongGuess)
        print("number of right guesses: ", self.rightGuess)
        print("accuracy: {}%".format(np.round(self.rightGuess*100/(self.wrongGuess+self.rightGuess),2)))
        

def split(X,Y,n):
    """splits data into training and testing sets"""
    X_train,X_test,y_train,y_test=model_selection.train_test_split(X.T,Y.T,test_size=n,random_state=1)
    X_train=X_train.T
    X_test=X_test.T
    y_train=y_train.T
    y_test=y_test.T
    return X_train,X_test,y_train,y_test

def initialize(n_x,n_h,n_y):
    "randomly initialize weights and bias"""
    #hidden layer
#    np.random.seed(0)
    W1=np.random.randn(n_h,n_x)*0.01
    B1=np.zeros((n_h,1))
    #output layer
#    np.random.seed(1)
    W2=np.random.randn(n_y,n_h)*0.01
    B2=np.zeros((n_y,1))
    return W1,B1,W2,B2
    
def forwardProp(W1,B1,W2,B2,X_train):
    """forward propgation"""        
    Z1=np.dot(W1,X_train)+B1
    A1=np.tanh(Z1)
    #output layer
    Z2=np.dot(W2,A1)+B2
    A2=1.0/(1+np.exp(-1*Z2))
    return Z1,A1,Z2,A2

def cost(A2,y_train,m):
    """cost"""
    a=y_train*np.log(A2)
    b=(1-y_train)*np.log(1-A2)
    cost=(-1/m)*(np.sum(a)+np.sum(b))
    return cost

def backProp(m,A2,W2,y_train,A1,X_train):
    """back propagation"""
    #output layer
    dZ2=A2-y_train
    dW2=(1.0/m)*np.dot(dZ2,A1.T)
    dB2=(1.0/m)*np.sum(dZ2,axis=1,keepdims=True)
    #hidden layer
    dZ1=W2.T*dZ2*(1-np.power(A1,2))
    dW1=(1.0/m)*np.dot(dZ1,X_train.T)
    dB1=(1.0/m)*np.sum(dZ1,axis=1,keepdims=True)
    return dW2,dB2,dW1,dB1

def update(W2,dW2,B2,dB2,W1,dW1,B1,dB1,alpha):
    """update"""
    W2_new=W2-alpha*dW2
    B2_new=B2-alpha*dB2
    W1_new=W1-alpha*dW1
    B1_new=B1-alpha*dB1
    return W2_new,B2_new,W1_new,B1_new


def plot(X,colors,error):
    fig, ax=plt.subplots(ncols=2)
    ax[0].scatter(X[0],X[1],c=colors)
    ax[1].plot(error)
    ax[1].set_title('cost per iteration')
    ax[1].set_xlabel('iteration')

    
    
    