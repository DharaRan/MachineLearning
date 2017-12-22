"""
Code by: Dhara Rana
Date: 9/18/17
Naives Bayes: Guassian Application

"""
import sys

#Opening Data and training labels
datafile=sys.argv[1] #"datasset.txt"
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


tdatafile=sys.argv[2] #"traininglabels.txt"
t=open(tdatafile,'r')

train={} #This is how you create a dictionary
l=t.readline()
while(l !=''):
    a=l.split()
    train[int(a[1])]=int(a[0])
    l=t.readline()
        

rows=len(data)
cols=len(data[0])

m0=[]
m1=[]
v0=[]
v1=[]
for i in range(0,cols):
    m0.append(1)
    m1.append(1)
    v0.append(1)
    v1.append(1)
n0=0
n1=0

#Calculate Mean
for i in range(0,rows):
    if(train.get(i)!= None and train[i]==0):
        n0 +=1
        for j in range(0,cols):
            m0[j] +=data[i][j]
    
    if(train.get(i)!= None and train[i]==1):
        n1 +=1
        for j in range(0,cols):
            m1[j] +=data[i][j]               
           
for j in range(0,cols):
    m0[j] /=n0
    m1[j] /=n1
    
#print(m0)
#print(m1)
n0=0
n1=0

#Calculate Variance
for i in range(0,rows):
    if(train.get(i)!= None and train[i]==0):
        n0 +=1
        for j in range(0,cols):
            v0[j] += (data[i][j]-m0[j])**2
    if(train.get(i)!= None and train[i]==1):
        n1 +=1
        for j in range(0,cols):
            v1[j] += (data[i][j]-m1[j])**2
 
for j in range(0,cols):
    v0[j] /=n0
    v1[j] /=n1
#print(v0)
#print(v1)    
               

 #Prediction

for i in range(0,rows,1):
    if(train.get(i)==None):
        d0=0
        for j in range(0,cols,1):
            d0+= ((data[i][j]-m0[j])**2)/v0[j]
        d1=0
        for j in range(0,cols,1):
            d1+= ((data[i][j]-m1[j])**2)/v1[j]
        if(d0<d1):
            print("0. ",i)
        else:
            print("1, ",i)
    
    
    