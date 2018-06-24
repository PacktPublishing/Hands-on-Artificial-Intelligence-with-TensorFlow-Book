'''
Created on 21-Jun-2018

@author: Ankit Dixit
'''

#We import pandas for data set operations
#Matplotlib to show plots
from matplotlib import pyplot as plt

#Numpy for algebric operations as well as, 
#for loading and storing text files
import numpy as np

#Seaborn to create plots
import seaborn as sns
 
#Load training data and labels from disk using numpy
X_tr = np.loadtxt('E:/HAPT Data Set/Train/X_train.txt',delimiter=' ').astype(np.float64)
Y_tr = np.loadtxt('E:/HAPT Data Set/Train/y_train.txt').astype(np.int32)
print('Size of Training Data: ',np.shape(X_tr))

'''
Size of Training Data:  (7767, 561)
'''

#Load test data and labels from disk using numpy
X_test = np.loadtxt('E:/HAPT Data Set/Test/X_test.txt',delimiter=' ').astype(np.float64)
Y_test = np.loadtxt('E:/HAPT Data Set/Test/y_test.txt').astype(np.int32)
print('Size of Testing Data: ',np.shape(X_test))

'''
Size of Testing Data:  (3162, 561)
'''

#Load Activity names
activity_labels = np.genfromtxt('E:/HAPT Data Set/activity_labels.txt',dtype='str')
print(activity_labels)

'''
[['1' 'WALKING']
 ['2' 'WALKING_UPSTAIRS']
 ['3' 'WALKING_DOWNSTAIRS']
 ['4' 'SITTING']
 ['5' 'STANDING']
 ['6' 'LAYING']
 ['7' 'STAND_TO_SIT']
 ['8' 'SIT_TO_STAND']
 ['9' 'SIT_TO_LIE']
 ['10' 'LIE_TO_SIT']
 ['11' 'STAND_TO_LIE']
 ['12' 'LIE_TO_STAND']]
'''

#Load Subject IDs
subject_ids = np.loadtxt('E:/HAPT Data Set/Train/subject_id_train.txt').astype(np.int64)

#Load Feature names
feature_names = np.genfromtxt('E:/HAPT Data Set/features.txt',dtype='str')
print('Feature Names: ',feature_names[:15])

'''
Feature Names:  ['tBodyAcc-Mean-1' 'tBodyAcc-Mean-2' 'tBodyAcc-Mean-3' 'tBodyAcc-STD-1'
 'tBodyAcc-STD-2' 'tBodyAcc-STD-3' 'tBodyAcc-Mad-1' 'tBodyAcc-Mad-2'
 'tBodyAcc-Mad-3' 'tBodyAcc-Max-1' 'tBodyAcc-Max-2' 'tBodyAcc-Max-3'
 'tBodyAcc-Min-1' 'tBodyAcc-Min-2' 'tBodyAcc-Min-3']
'''


#Let's see classs distributions for all activities
ax = sns.countplot(Y_tr,label="Count")
plt.show()

#We will pick subject number 5 to see variation in different activity
#for first two features

#Lets extract indices of subject id
sub_id = 5
ids = np.where(subject_ids==sub_id)[0]

#Extract rows for subject_id from training data and labels
subject_10_X = X_tr[ids[0]:ids[-1],:]
subject_10_Y = Y_tr[ids[0]:ids[-1]] 
print('Number of samples of subject: ',np.shape(subject_10_X))

'''
Number of samples of subject:  (323, 561)
'''
#Set plot context for font size 
sns.set_context("paper", rc={"font.size":1,"axes.titlesize":1,"axes.labelsize":1})

#Let's plot value variation of 2 features for each activity
fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(211)
ax1 = sns.stripplot(x=subject_10_Y, y = subject_10_X[:,0], jitter=True)
ax1.set(xticklabels=activity_labels)

ax2 = fig.add_subplot(212)
ax2 = sns.stripplot(x=subject_10_Y, y = subject_10_X[:,1], jitter=True)
ax2.set(xticklabels=activity_labels)
plt.show()