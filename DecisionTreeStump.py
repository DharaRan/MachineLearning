# -*- coding: utf-8 -*-
"""
Dhara Rana
UCID: djr32
Hw 5: Decision Tree: Creating Stump of Split

"""
import sys
def classes(dataset,labelCol):
    """
    Finds the unique values for a column in a dataset
    Ex. 0 and 1
    """
    return list(set(row[labelCol] for row in dataset))

def split(thres,coln,dataset):
   """
   Split data left and right depending on the threshold set 
   by value= derived from dataset and coln.
   If value < thres, left side
   If value > thres, right side
   """
   left=list()
   right=list()
   
   for row in dataset:
       if row[coln]<thres:
           left.append(row)
       else:
           right.append(row)
   return left,right # returns the 2 groups

def gini_index(groups,classes):
    """
    Calculates the gini index
    """
    left=groups[0]
    right=groups[1]
    
    tot_rows=len(left)+len(right)
    gini=0.0
    for group in groups:
        size=len(group)
        #print("Group is ", str(group))
        if size == 0:
            continue
        prob=1
        for class_val in classes:
            p=[row[-1] for row in group].count(class_val)/size
           # print("Row: ", str([row[-1] for row in group]))
            #print("For class ",str(class_val),"p is ",str(p))
            prob=prob*p
        gini +=(prob)*(size/tot_rows)
    return gini
        
    
def get_split(dataset,labelCol):
    classval= classes(dataset,labelCol)
    b_coln=0
    b_row=0
    b_value=0
    b_gini=1 
    b_groups= None
    sim_count=0
    for col in range(len(dataset[0])-1):
        for row in range(len(dataset)):
            groups=split(dataset[row][col],col,dataset)
            gini=gini_index(groups,classval)
            #print('X%d < %.3f Gini=%.3f' % ((col+1), dataset[row][col], gini))
            if gini < b_gini:
                b_coln=col
                b_row=row
                b_value=dataset[row][col]
                b_gini=gini
                b_groups=groups
            elif gini==b_gini:
                sim_count= sim_count+1
    #print('Total similar count',str(sim_count))
    if(sim_count==((len(dataset)*2)-1)):
        b_coln=0
        #row value is going to be the max in the column 0  
        b_rowVal=dataset[0][b_coln]
        b_row=0
        for row in range(len(dataset)):
            if dataset[row][b_coln]>b_rowVal:
                b_row=row
                b_rowVal=dataset[row][b_coln]
        b_value=dataset[b_row][b_coln]
        b_gini=gini
        b_groups=split(dataset[b_row][b_coln],b_coln,dataset)      
                                
                
                
    return {'column':b_coln,'row':b_row, 'value':b_value, 'groups':b_groups,'gini':b_gini}
    
def getSplitLine(b_coln,b_value,dataset):
    #print('b_value: ', str(b_value))
    win_col=list()
    maxNum=-9999# some very small number
    for r in range(len(dataset)):
        win_col.append(dataset[r][b_coln])
    win_col.sort()
    for r in range(len(dataset)):
        val=dataset[r][b_coln]
        if val<b_value:
            if val>maxNum:
                maxNum=val
                #print(maxNum)
            
    s=(maxNum+b_value)/2
    return s
    

datafile= sys.argv[1] #"datasset6.txt" #
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


tdatafile= sys.argv[2] #"traininglabels6.txt" #
t=open(tdatafile,'r')
label={} #This is how you create a dictionary
l=t.readline()
while(l !=''):
    a=l.split()
    label[int(a[1])]=int(a[0])
    l=t.readline()

pred=list()
#Merge Traindata and labels and remove pred values
for r in range(len(data)):
    if (label.get(r)!= None):
        data[r].append(label[r])
    else:
        pred.append(data[r])
dataset=list()
for r in data:
    length= len(r)
    if length==len(data[0]):
        dataset.append(r)
    
labelCol=len(dataset[0])-1
stump = get_split(dataset,labelCol)
#print('Split Threshold: [X%d < %.3f]' % ((stump['column']+1), stump['value']))  
s=getSplitLine(stump['column'],stump['value'],dataset)

#print('K (column): ',str(stump['column']),' s: ', str(s),'gini: ',str(stump['gini']))

print('Best column: %d' %stump['column'])

print('Gini is: %f'%stump['gini'])

print('Split point value: %f' %s)    
    


