
"""
Code by: Dhara Rana (djr32)
Date: 9/25/17
Gradient Descent with  Least Squares Loss
"""
import sys
import random
import math
# This calculates dot product

def DotProduct(w,data):
    dp = [x * y for x, y in zip(w, data)]
    return sum(dp)

#Opening Data and training labels
datafile= "datasset.txt" #sys.argv[1] 
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


tdatafile= "traininglabels.txt" #sys.argv[2]
t=open(tdatafile,'r')

train={} #This is how you create a dictionary
l=t.readline()
while(l !=''):
    a=l.split()
    train[int(a[1])]=int(a[0])
    l=t.readline()
        

rows=len(data)
cols=len(data[0])
for i in train:
    if train[i] == 0:
        train[i]=-1

for i in range(0,len(data)):
    data[i].append(1)
    
rows=len(data)
cols=len(data[0])

# Step1: Initialize inital w
w=[]

for i in range(0,cols):
    #w.append(random.uniform(0, 0.01))
    #w.append(random.randint(0,1))
    w.append(0.02 * random.random() - 0.01)
    

#w0=w[0]
#w1=w[1]


#h(x)= w1x1+ w2x2+w0 assume no bais
#Step 2: Calculate inital total error, J(theta)
J=0
for i in range(0,rows):
    if(train.get(i)!= None):
       J += (train[i]-DotProduct(w,data[i]))**2
#print('Print intial J ' + str(J))

#Step 2: Set your learning rate
eta=0.001 #t learning rate

#Step 3: Gradiant decesent 
#hypothesis is h(x)=Wo+W1x (x is the test point)
converged=False;
dellf=[0]*cols
temp1=[[0,0,0]]*rows
dp1=[0]*rows
grad=[0]*cols
h=[0]*cols
pdict1=[0]*rows

while not converged:
   
    #Step 3a: Find the h(x)-y; 
    #h(x) is found by taking the dot product of w and data[i]
    for i in range(0,len(data)):
        if(train.get(i)!= None):
            pdict1[i] = DotProduct(w, data[i]) - train[i]
            
    #Step3b: Find the multiplying each (h(x)-y) by the each data point
    #(h(x)-y^i)x^i
    #This will result in a list of 8 rows and 3 columns
    for i in range(0,len(data)):
        if(train.get(i)!= None):
            for j in range(0,len(data[0])):
                h[j]=(data[i][j]*pdict1[i])
            temp1[i]=h
            h=[0]*3
    
    #Step 3c: New weights; find the gradiant; take the sum of the values in each column  
    grad =  [sum(l) for l in zip(*temp1)]

    #Step3d: update w
    for k in range(0,cols):
        w[k]= w[k]- eta*grad[k]
    #Step 3e: Find new error with new weight 
    error=0
    for t in range(0,rows):
        if(train.get(t)!= None):
           error += (train[t]-DotProduct(w,data[t]))**2
           
    #print('Error in loop ' + str(error))
    
    #Step 3f: Check previous error with new error if its less than convergence number
    if abs(J-error) <= 0.001:
       converged = True
       print("We have converged!!!")
    diffError=abs(J-error)
    #print("Diff between J-error "+ str(diffError))
    
    #Step 3g: make the new error equal the previous error
    J=error #update J(theta)


print('W0 is ' + str(w[0]))
print('W1 is ' + str(w[1]))

normw=0;
for i in range(0,cols-1):
    normw += w[i]**2

#print("normal weight " + str(normw))
normw=math.sqrt(normw)

#h(x)=w0x1+w1x2+w3(1)
d_origin =abs(w[len(w)-1]/normw)
print('The origin is ' +str(d_origin))

#Prediction
for i in range(0,rows):
    if(train.get(i)== None):
        print('Test point ' + str(data[i]))
        #print('w is ' + str(w))
        dp=DotProduct(w,data[i])
        print('dp ' + str(dp))
        if(dp>0):
            print('1')
        else:
            print('0')
    














