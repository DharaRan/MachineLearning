# -*- coding: utf-8 -*-
"""
Dhara Rana
UCID: djr32
Due Date: 9 Nov 2017
Hw 6: BootStrapping with Decision Stump

"""


import sys
import random
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
           # print('X%d < %.3f Gini=%.3f' % ((col+1), dataset[row][col], gini))
            if gini < b_gini:
                b_coln=col
                b_row=row
                b_value=dataset[row][col]
                b_gini=gini
                b_groups=groups
            elif gini==b_gini:
                sim_count= sim_count+1
   # print('Total similar count',str(sim_count))
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
                
    return {'column':b_coln,'row':b_row, 'value':b_value, 'groups':b_groups}
    
def getSplitLine(b_coln,b_value,dataset):
    #print('b_value: ', str(b_value))
    win_col=list()
    maxNum=-9999# some very small number
    for r in range(len(dataset)):
        win_col.append(dataset[r][b_coln])
    win_col.sort()
    for r in range(len(dataset)):
        val=dataset[r][b_coln]
        #print('val ', str(val))
        if val<b_value:
            if val>maxNum:
                maxNum=val
                #print(maxNum)
            
    s=(maxNum+b_value)/2
    return s
def ClassSide(best_col,best_split,labelCol,dataset):
    """
    Determine which side is 0 and which side is -1
    """
    classval= classes(dataset,labelCol)
    if(len(classval)==1):
        if(classval[0]==1):
            classval.append(0)
        else:
            classval.append(1)
    classval.sort()
    """
    Count the number of each label on each side
    """
    sideA=0
    sideB=0
    for r in range(0,len(dataset)):
        if(dataset[r][best_col] <best_split):#on the left side
            if(dataset[r][labelCol]==classval[0]):
                sideA=sideA+1
            else:
                sideB=sideB+1
        if(sideA>sideB):
            left=classval[0]
            right=classval[1]
        else:
            left=classval[1]
            right=classval[0]
                
    
    return{'left':left,'right':right}

def Stump_predict(best_col,best_split,newValue,leftClass,rightClass):
    """
    Predict if the 
    new value< best_split 0
    new value >= best_split 1
    """
    #print("Winner Column used for testpoint: ", str(best_col))
    classification=0;
    #predict=list()
    for row in [newValue]:
        if row[best_col] < best_split:
            #row.append(classval[1])
            classification=leftClass
        
        else:
            #row.append(classval[0])
            classification=rightClass

    return classification

"""
Main body of code
"""
"""
Get Data and labels
"""
datafile= sys.argv[1] #"ionosphere.data"#"datasset6.txt" #
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

tdatafile= sys.argv[2] #"ionosphereLabels.labels"#"traininglabels6.txt" #
t=open(tdatafile,'r')
label={} #This is how you create a dictionary
l=t.readline()
while(l !=''):
    a=l.split()
    label[int(a[1])]=int(a[0])
    l=t.readline()

test=list()
#Merge Traindata and labels and remove pred values
for r in range(len(data)):
    if (label.get(r)!= None):
        data[r].append(label[r])
    else:
        test.append(data[r])
dataset=list()
for r in data:
    length= len(r)
    if length==len(data[0]):
        dataset.append(r)
   
"""
Bootstrapping begins
"""
ncols=len(dataset[0])-1             # Number of feature columns
nrows=len(dataset)                  # Number of rows in data
labelCol=len(dataset[0])-1       # Column number of the labels
ratio=1/3                        # ratio of column selection
numBootCol=round((ncols+1)*ratio) 
classval= classes(dataset,labelCol)
classval.sort()

for test_pt in range(len(test)):
# start of big outer loop here for BootStrapping
    #print("setting m1-0 and m=0")
    countm1=0
    count1=0
    
    for k in range(0,100): # bootstraaping!!
        
        """
        Select random numBoot cols and nrows from are real dataset
        """
        ranCol=random.sample(range(0, ncols),ncols)
        ranCol.append(labelCol)     #After random column is selected, the label column append at the end
        # print('rancol: ', str(ranCol))
        ranRow=list()
        for row in range(nrows):
            ranRow.append(random.randint(0, nrows-1))
        #print('ranRow: ', str(ranRow))
        """
        Create bootstrap dataset called b_data
        """
        b_data=[0]*len(ranRow)
        temp=[0]*len(ranCol)
        i=0
        j=0
        for r in ranRow: 
            for c in ranCol:
                temp[i]=dataset[r][c]
                i=i+1
            b_data[j]=temp
            temp=[0]*len(ranCol)
            i=0
            j=j+1
        #print('b_data', str(b_data))
                
        """
        Descesion Stump Classification
        """
        labelColb=len(b_data[0])-1       # Column number of the labels
        stump = get_split(b_data,labelColb)
        #print('Split Threshold: [X%d < %.3f]' % ((stump['column']+1), stump['value']))

        """
        Set which side equals 1 or 0
        """
        cs=ClassSide(stump['column'],stump['value'],labelColb,b_data)   
        """
        Equvient the 0 column in b_data to the actually in teh dataset and testpoint
        example: teh randomcol (rancol) we selected in the beginning is was 1,3 (3 is the label col so disregard)
        Column 0 in b_data is equal to column 1 in the real dataset
        """
        best_column=stump['column'] #in b_data
        b_column_data=ranCol[best_column] # best column in the real dataset            
        """
        Prediction
        """
        predict=Stump_predict(b_column_data,stump['value'],test[test_pt],cs['left'],cs['right'])

        #print('Prediction: ', str(predict))
            
        """
         Keep count of majoirty vote
        """
        #predVal=predict[0][len(predict[0])-1]
        if predict== classval[0]:
           countm1=countm1+1
        else:
           count1=count1+1
    
    #end of Bootstrappingloop
    """
    Final predication
    """
    #print('Total majority classification for ',str(classval[0]),": ", str(countm1))
    #print('Total majority classification for ',str(classval[1]),": ", str(count1))
    
    if countm1>count1:
        final_predict=classval[0]
        print('Final prediction for ',str(test[test_pt]),':', str(final_predict))
    else:
        final_predict=classval[1]
        print('Final prediction for ',str(test[test_pt]),':', str(final_predict))

#End of iteration through multiple Testpoints Loop










   