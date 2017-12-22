# -*- coding: utf-8 -*-
import sys
import numpy as np
import math
"""
Code by: Dhara Rana
Date: 9/16/17
Naives Bayes: Guassian Application

"""
"""
Mean function finds the mean
Input is the data inputted as array of strings
"""
def mean(data):
    total=0
    for m in data:
        total=total+int(m)
       
    average=total/len(data)    
    return(average)

"""
var function finds the variance- specifically-population variance
Input is the data inputted as array of strings
"""
def var(data):
    average=mean(data)
    total=0
    for m in data:
        different=int(m)-average
        total=total+ (different**2)
    
    var=total/len(data)    
    return(var)

"""
liklihood function 
find the liklihood of a specific have a certain classification
Ex. the probalitliy that feature 1 will be classified as 0.
This liklihood is represented by the Gaussian distribution with the 
parameters mean and variance
inputs: (1) attribute of interest to be classifed (e.g feature1=9 or height=90cm)
        (2)mean of attribute of interest and class(3) variance of attribute of 
        interest and class
 output: the likihood; 
 e,g P(f1=3|class=0)   

"""
def liklihood(attribute,mean,var):
    f1x=int(attribute)# Attribute is the instance or new data set of one attribute of table like feature2=3
    x=(-1/2)*((f1x-mean)**2/var)
    liklihood = math.exp( x )/math.sqrt(2*math.pi*var)
    return liklihood


"""
findDataClassFeature function 
Returns a list of strings where in the data a specific class occurs 
of a specific attributes. So all the data where class= 0  for the feature1 
input: (1) the data set 
(2) the indices where the interest class occures. E.g all the indices where class=0 from the training labels
(3) featureCol: the index in the data set that represented of desired data 
E.g In a dataset of height and weights, height has a index of 0. for featureCol=0
 Output: is a list of data


"""
def findDataClassFeature(data,indexList, featureCol):
    f1c0=[]
    for j in range(0,len(indexList)):
        f1c0.append(data[int(indexList[j]),featureCol]) 
    return f1c0
"""
GuassianNaiveBayes function 
input: (1) data set (array)(2) training labels-array (format:classification,indices)
       (3) target value in feature1 () target value in feature 2
Output: 
    0 or 1
"""
def GuassianNaiveBayes(data,train,f1,f2):
    """
     First find the Guassian for each type of classification with respect to each 
     feature.
     Ex. For a data set with height and weight where the two attributes tells us whether
     its a adult or child. 
     We would have 4 sets the arithmetic mean and variance (using pop. var) of each 
     attributes with each classifier
     So mean and variance for  (1)height, adult (2) weight,adult (3) height, child
     (4) weight, child
    """            
    indices0=[]
    indices1=[]
    #Find the indices where class=0
    for index0 in range(0,len(train)):
        a=train[index0,0]
        if a =='0':
            i=train[index0,1]
            indices0.append(i)
    #print("All indices where class=0: "+ str(indices0))
           
    #Find the indices where class=1
    for index1 in range(0,len(train)):
        a=train[index1,0]
        if a =='1':
            i=train[index1,1]
            indices1.append(i) 
    #print("All indices where class=1: "+ str(indices1))
    
    #Find all Feature 1 where class=0
    f1c0 =findDataClassFeature(data,indices0, 0)
    #print("Feature 1 where class=0: "+ str(f1c0))
    meanf1c0=mean(f1c0) # find mean
    varf1c0=var(f1c0)#Find the variance of Feature 1 where class=0
    #Find all Feature 2 where class=0
    f2c0 =findDataClassFeature(data,indices0, 1)
    #print("Feature 2 where class=0: "+ str(f2c0))
    meanf2c0=mean(f2c0) # find mean
    varf2c0=var(f2c0) #Find the variance of Feature 1 where class=1
    
    #Find all Feature 1 where class=1
    f1c1 =findDataClassFeature(data,indices1, 0)
    #print("Feature 1 where class=1: "+ str(f1c1))
    meanf1c1=mean(f1c1)# find mean
    varf1c1=var(f1c1) #Find the variance of Feature 2 where class=0
    
    #Find all Feature 2 where class=1
    f2c1 =findDataClassFeature(data,indices1, 1)
    #print("Feature 3 where class=1: "+ str(f2c1))
    meanf2c1=mean(f2c1) # find mean
    varf2c1=var(f2c1) #Find the variance of Feature 2 where class=1
  
    mean_var=[[meanf1c0,varf1c0],[meanf2c0,varf2c0],[meanf1c1,varf1c1],[meanf2c1,varf2c1]]
    #print(mean_var)
    
    mean_var=np.array(mean_var)
    
    # Find prior probabilities of class =0 and class=1
    pclass0=len(indices0)/len(train) #P(class=0)
    #print("P(class=0): " +str(pclass0))
    pclass1=len(indices1)/len(train) #P(class=1)
    #print("P(class=1): " +str(pclass1))
    
    #Find all the likilhoods; there are 4
    probf1c0=liklihood(f1,meanf1c0,varf1c0)#Likihoold of P(F1=1|class=0)
    #print("P(f1=1|class=0): " +str(probf1c0))
    probf2c0=liklihood(f2,meanf2c0,varf2c0)#Likihoold of P(F2=3|class=0)
    #print("P(F2=3|class=0): " +str(probf2c0))
    probf1c1=liklihood(f1,meanf1c1,varf1c1)#Likihoold of P(F1=1|class=1)
    #print("P(F1=1|class=1): " +str(probf1c1))
    probf2c1=liklihood(f2,meanf2c1,varf2c1)#Likihoold of P(F2=3|class=1)
    #print("P(F2=3|class=1): " +str(probf2c1))
    
    #Normalization
    #X= Xf1=1 &Xf2=3
    probX1_3c0=probf1c0*probf2c0 #P(X|class=0)=P(f1=1|class=0) *P(f2=3|class=0)
    #print("P(X|class=0): " +str(probX1_3c0))
    probX1_3c1=probf1c1*probf2c1 #P(X|class=1)=P(f1=1|class=1) *P(f2=3|class=1)
    #print("P(X|class=1): "+str(probX1_3c1))
    
    #Bayes theorem applied
    Pclass0_X=(probX1_3c0*pclass0)/((probX1_3c0*pclass0)+(probX1_3c1*pclass1))
    #print("P(class=0|X): "+str(Pclass0_X))
    Pclass1_X=(probX1_3c1*pclass1)/((probX1_3c1*pclass1)+(probX1_3c0*pclass0))
    #print("P(class=1|X): "+str(Pclass1_X))
    
    #Classification
    if Pclass0_X>Pclass1_X:
        classified=0
    else:
         classified=1
    
    return classified


#Opening Data and training labels
datafile=sys.argv[1]
data=[]

with open(datafile) as f:
    # Iterate through the file until the table starts
    for line in f:
        if line.startswith('feature1'):
            break
    # Read the rest of the data, using spaces to split. 
    data = [r.split() for r in f]

datafile1=sys.argv[2]
train=[]
with open(datafile1) as a:
    # Iterate through the file until the table starts
    for line1 in a:
        if line1.startswith('class'):
            break
    # Read the rest of the data, using spaces to split. 
    train = [r.split() for r in a] 
    data=np.array(data)
    train=np.array(train)


#Find all the missing classification
missingClasslen=len(data)-len(train)
missingClass=[]
counter=len(train)
for i in range(0,missingClasslen):
    missingClass.append(data[counter])
    counter=counter+1
missingClass=np.array(missingClass)

missClassified=[[],[]]
for h in range(len(missingClass)):    
    ans=np.where((data[:,0]==missingClass[h,0]) & (data[:,1]==missingClass[h,1]))
    indices=ans[0][0]
    classifies=GuassianNaiveBayes(data,train,missingClass[h,0],missingClass[h,1])
    missClassified[h]=[classifies,indices]
    
print("Classification of missing data (class,indices): "+ str(missClassified))







