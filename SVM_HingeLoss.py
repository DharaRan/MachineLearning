
"""
Code by: Dhara Rana (djr32)
Date:10/5/17
SVM with Hinge Loss
"""
import sys
import random
import math

# This function calculates dot product
def DotProduct(w,data):
    dp = [x * y for x, y in zip(w, data)]
    return sum(dp)

#Opening Data and training labels
datafile= "datasset2.txt" #sys.argv[1] 
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


tdatafile= "traininglabels2.txt" #sys.argv[2] 
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
count=0
for i in range(0,rows):
    if(train.get(i)== None):
        count+=1


# Initialize some w
w=[]
for i in range(0,cols):
    w.append(0.02 * random.random() - 0.01)

#H(x)= W'Xi
# total error, J(theta)=summation of max(0,1-yi(H(xi)))
J=0
for i in range(0,rows):
    if(train.get(i)!= None):
       #a=Yi*(W^t*Xi)
       v=train[i]*DotProduct(w,data[i])
       J += max(0,1-v)
#print('Print intial J ' + str(J))

eta=0.001 #this is learning rate

#Gradiant decesent 
#hypothesis is h(x)=Wo+W1x (x is the test point)
converged=False;
grad=[0]*cols
h=[0]*cols
pdict=[0]*(rows-count)
while not converged:
   
    #Find sub-gradient 
    #find yi(W'Xi)
    for i in range(0,rows):
        if(train.get(i)!= None):
            pdict[i]=train[i]*DotProduct(w,data[i])
    #Let's check if each predict is etiher >1 or <1, which will give us the gradient
    dellf=[0]*(rows-count)
    for i in range(0,rows):
        if(train.get(i)!= None):
            if (pdict[i] <1):
                for j in range(0,cols):
                    h[j]=-1*train[i]*data[i][j]
                dellf[i]=h
                h=[0]*cols
            else:
                for j in range(0,cols):
                    h[j]=0
                dellf[i]=h
                h=[0]*cols
    #Sum up the gradient values or each column
    grad =  [sum(t) for t in zip(*dellf)]

    #update w
    for k in range(0,cols):
        w[k]= w[k]- eta*(grad[k])
    error=0
    for i in range(0,rows):
        if(train.get(i)!= None):
           a=train[i]*DotProduct(w,data[i])
           error += max(0,1-a)
           
    
    if abs(J-error) <= 0.000000001:
       converged = True
       #print("We have converged!!!")
    #diffError=abs(J-error)
    #print("Diff between J-error "+ str(diffError))
    J=error #update J(theta)

print('W is ' + str(w[0])+","+str(w[1]))
print('W0 is ' + str(w[2]))
normw=0;
for i in range(0,cols-1):
    normw += w[i]**2

#print("normal weight " + str(normw))
normw=math.sqrt(normw)

#h(x)=w1x1+w2x2+w0(1)
d_origin =abs(w[len(w)-1]/normw)
print('The distance from origin is ' +str(d_origin))

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
    














