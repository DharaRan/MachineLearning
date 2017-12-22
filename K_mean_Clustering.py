"""
Dhara Rana
UCID: djr32
21 November 2017
HW 7: Implement K-mean Clustering
"""
import sys
import random
def distance(centroid, datapoint):
    """
    Calulate the distance between one centroid 
    and a data point ; both are same length
    """
    d=0
    for j in range(len(datapoint)-1):
        d=d+(datapoint[j]-centroid[j])**2
    d= d**(1/2)
    return d

def has_converged(oldMean,newMean):
    return(set([tuple(a) for a in oldMean])==set([tuple(a) for a in newMean]))


datafile= sys.argv[1] # "datasset7.txt" #
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

K=int(sys.argv[2])#2#int(input("Enter number of clusters: "))#

# Add the index column to the labels. It will always be the last column

for index in range(len(data)):
    data[index].append(index)

"""
Initalize centroid as two datapoint from data set;
"""
centroid=random.sample(data, K)

converged=False
while converged==False:
    """
    Cluster Assignment step
    """
    dist=[]*K
    
    for row in data:
        temp=list()
        for c in range(0,K):
            temp.append(distance(centroid[c],row))
        dist.append(temp)
    
    #Find the minimium distance between is centroid
    #Assign it into a group
    category=[ [] for _ in range(K)]
    for r in range(len(dist)):
        clusterNum=dist[r].index(min(dist[r]))#which cluster Num has the smallest distance from the rest of the clusters
        for clus in range(0,K):
            if clusterNum == clus:
                category[clus].append(data[r])   
    """
    Move Centroid step
    """
    # find mean of each cluster group
    NewCentroid=[ [] for _ in range(K)]
    for clusterGroup in range(len(category)):#the cluster group
        for col in range(len(category[0][0])-1):
            sumC=0        
            for row in range(len(category[clusterGroup])):
                sumC=sumC+category[clusterGroup][row][col]
            
            m=sumC/len(category[clusterGroup])
            NewCentroid[clusterGroup].append(m)
    """
    Check for convergences otherwise update
    """        
    converged=has_converged(centroid,NewCentroid)
    if converged==True:
        f=list()
        print("We have converged")
        for cat in range(len(category)):
            for row in range(len(category[cat])):
                indexCol=len(category[0][0])-1
                f.append([cat,category[cat][row][indexCol]])
                #print(str(cat)," ",str(category[cat][row][indexCol]))
        f.sort(key=lambda x: x[1])
        for r in range(len(f)):
            print(str(f[r][0])," ",str(f[r][1]))
        centroid=NewCentroid
        for cluster in range(len(centroid)):
            print('Cluster',str(cluster),': ', str(centroid[cluster]))
            
    centroid=NewCentroid





















