"""
Code by: Dhara Rana (djr32)
Date: 10/16/17
Logistic Regression
"""
import sys
import random
import math


def DotProduct(w,data):
    dp = [x * y for x, y in zip(w, data)]
    return sum(dp)
def sigmoid(w,dataRow):
    dp=DotProduct(w,dataRow)
    sig=1.00/(1.00+math.exp(-dp))
    if(sig >= 1.0):
        sig = 0.999999
    
    return sig



#Open Data File
datafile= "datasset4.txt" #sys.argv[1] #
print(datafile)
f=open(datafile,'r')
data=[]
i=0
l=f.readline()
while(l !=''):
    a=l.split()
    l2=[]
    for j in range(0,len(a),1):
        l2.append(float(a[j]))
    data.append(l2)
    l=f.readline()
    
#append a one vector
for i in range(0,len(data)):
    data[i].append(1)

#Open Training Labels Data file
tdatafile= "traininglabels4.txt" #sys.argv[2] #
t=open(tdatafile,'r')

train={} #This is how you create a dictionary
l=t.readline()
while(l !=''):
    a=l.split()
    train[int(a[1])]=int(a[0])
    l=t.readline()

rows=len(data)
cols=len(data[0])
print("Please wait...Code is Running.")
# Initialize some w
w=[]
for i in range(0,cols):
    #w.append(random.uniform(0, 0.01))
    w.append(0.02 * random.random() - 0.01)

#Cacluate the inital Cost Function with the inital w
J=0
for i in range(0,rows):
    if(train.get(i)!= None):
        #J+= (train[i] - math.log(1 + math.exp(-1 * DotProduct(w, data[i]))))**2
       J+=(train[i]*math.log(sigmoid(w,data[i]))+((1-train[i])*math.log(1-sigmoid(w,data[i]))))*-1
#print('Print intial J ' + str(J))

#Calcuate the gradient and converge to find w for dataset

converged=False;
dellf=[0]*cols
temp=[[0,0,0]]*rows
grad=[0]*cols
h=[0]*cols
pdict=[0]*rows
eta=0.01

while not converged:
    #Calculate h(x)-y
    #This results in a 1 column by n rows
    for i in range(0,rows):
        if(train.get(i)!= None):        
            pdict[i]= sigmoid(w,data[i])-train[i]
        
        
    #Calculate (h(x)-y)x
    #This will result in n rows by m cols
    
    for i in range(0,rows):
        if(train.get(i)!= None):
            for j in range(0,cols):
                h[j]=data[i][j]*pdict[i]
            temp[i]=h
            h=[0]*3
    
    #Find the sum of each column
    grad =  [sum(t) for t in zip(*temp)]

    #update w
    for k in range(0,cols):
        w[k]= w[k]- eta*grad[k]
    
    #Find error using new w
    error=0
    for i in range(0,rows):
        if(train.get(i)!= None):
           v=sigmoid(w,data[i])
           #print("v is " +str(v))
           error += (train[i]*math.log(sigmoid(w,data[i]))+((1-train[i])*math.log(1-sigmoid(w,data[i]))))*-1
           #error += (train[i] - math.log(1 + math.exp(-1 * DotProduct(w, data[i]))))**2
    #print('Error in loop ' + str(error))
    
    if abs(J-error) <= 0.0000001:
       converged = True
       #print("We have converged!!!")
    #print("Difference b/w obj: " + str(abs(J-error)))   
    J=error #update J(theta)
    
print('W1 is ' + str(w[0]))
print('W2 is ' + str(w[1]))

normw=0;
for i in range(0,cols-1):
    normw += w[i]**2


normw=math.sqrt(normw)
print("||w||: " + str(normw))
#h(x)=w0x1+w1x2+w3(1)
d_origin =w[len(w)-1]/normw
print('The distance from origin is ' +str(d_origin))


#Prediction
for i in range(0,rows):
    if(train.get(i)== None):
        print('Test point ' + str(data[i]))
        #print('w is ' + str(w))
        dp=DotProduct(w,data[i])
        #print('dp ' + str(dp))
        if(dp>0):
            print('1')
        else:
            print('0')






