
"""
Dhara Rana
UCID: djr32

Random Hyper Plane Assignment

"""
import random
import math
import sys
from sklearn import svm
from sklearn.model_selection import cross_val_score
import os

def DotProduct(w,data):
    dp = [x * y for x, y in zip(w, data)]
    return sum(dp)


datafile= sys.argv[1] #"breastcancer.txt" #
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



datafile= sys.argv[2]#"breast_cancerLab.txt" #
f=open(datafile,'r')
label=[]
i=0
l=f.readline()
while(l !=''):
    a=l.split()
    l2=[]
    for j in range(0,len(a),1):
        l2.append(float(a[j]))
    label.append(l2)
    l=f.readline()

del(datafile,l,l2,a,i,j)

trainLabels,index=zip(*label)
trainLabels=list(trainLabels)
del(index)

#testA = sys.argv[3]

rows=len(data)
cols=len(data[0])

planes=[10, 100,1000,10000]
file_path = os.path.dirname(os.path.abspath('__file__'))
path=label_path = file_path+'\\errorReport'
f = open(path, "w")


for k in planes:
    newFeatureVec=[]
    f.write("Number of Planes: ")
    f.write(str(k) +"\n")
    print('Number of Planes:',k)
    for ktimes in range(0,k,1):
        #create random hyperplane
        w=[]
        for i in range(0,cols):
            #w.append(random.uniform(0, 0.01))
            w.append(2 * random.random() - 1)
        w.append(0)
        #Create new feature Vector
        hyperplane=[]
        for i in range(0,rows):
            dp=DotProduct(w,data[i])
            #(1+sign(zi))/2
            sign=int(math.copysign(1,dp))
            val=int(1+sign)/2
            hyperplane.append(val)
        newFeatureVec.append(hyperplane)

    #This transformed Matrix
    newFeatureVec_t=zip(*newFeatureVec)
    traindata = []
    for row in newFeatureVec_t:
        traindata.append(row)

    
    model = svm.LinearSVC(C=0.01)
    scores = cross_val_score(model, traindata, trainLabels, cv=5) # fitting a model on hyperPlane Data
    scores[:] = [1-x for x in scores]
    scoresMean=scores.mean()
    scores_orig = cross_val_score(model, data, trainLabels, cv=5) # fitting a model on Original  Data
    scores_orig[:] = [1-x for x in scores_orig]
    scoresOrigMean=scores_orig.mean()
    
    print("Error for new features data: ",scores)
    print("Mean for new features data: ",scoresMean)
    print("Error for orginal data: ",scores_orig)
    print("Mean for  orginal data: ",scoresOrigMean)
    
    f.write("Error for new features data: "+ "\n")
    f.write(str(scores) +"\n")
    f.write("Mean for new features data: "+ "\n")
    f.write(str(scoresMean) +"\n")
    f.write("Error for orginal data: "+ "\n")
    f.write(str(scores_orig) +"\n")
    f.write("Mean for orginal data: "+ "\n")
    f.write(str(scoresOrigMean) +"\n\n")
f.close()

print('Prediction file Saved in: ',path)




if len(sys.argv)==4:    
    datafile= sys.argv[3]#"testdata" # #Feature selected Train data
    f=open(datafile,'r')
    realtest=[]
    i=0
    l=f.readline()
    while(l !=''):
        a=l.split()
        l2=[]
        for j in range(0,len(a),1):
            l2.append(float(a[j]))
        realtest.append(l2)
        l=f.readline()
    del(datafile,l,l2,a,i,j)
    print('thrid file read')
    prediction=model.predict(realtest)
    print('Prediction of TestData: ')
    print(prediction)
    



