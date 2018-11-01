'''
Created on 15-Jun-2018

@author: Ankit Dixit
'''

#Let's start with downloading and load our data set in matrix form

#Import pandas to manage data set 
import pandas as pd

#Import Tensorflow to get the file from given URL 
import tensorflow as tf

#Import NumPy for all mathematics operations on numerical data
import numpy as np

#Here's the name we'll give our data
data_name = 'wsbc.csv'

#Here's where we'll load the data from our github repo
data_url = 'https://github.com/PacktPublishing/'+\
           'Hands-on-Artificial-Intelligence-with-TensorFlow-Book/'+\
           'blob/master/Chapter_10/Data/wdbc.csv'

#Let's load the data from URL
file_name = tf.keras.utils.get_file(data_name, data_url)

#Load into a variable using pandas read_csv
data = pd.read_csv(file_name, delimiter=',')

#Let's verify the size of data set
print('Number of Instances: %d\nNumber of attributes: %d'%(data.shape[0],data.shape[1]))

'''
Number of Instances: 569
Number of attributes: 33
'''

#Let's take a look name of all features
print(data.columns)

'''
Index(['id', 'diagnosis', 'diagnosis_numeric', 'radius', 'texture',
       'perimeter', 'area', 'smoothness', 'compactness', 'concavity',
       'concave_points', 'symmetry', 'fractal_dimension', 'radius_se',
       'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worse', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave_points_worst',
       'symmetry_worst', 'fractal_dimension_worst'],
      dtype='object')
'''

#Now we will select some features and see their values 
features = ['radius','texture','smoothness', 'compactness', 'concavity','diagnosis']

#We will shuffle the data set before visualization
data = data.reindex(np.random.permutation(data.index))
data_train_test = data.copy()
print(data.head(10)[features])

'''
      radius  texture  smoothness  compactness  concavity diagnosis
36    12.18    20.52     0.08013      0.04038    0.02383         B
223   13.51    18.89     0.10590      0.11470    0.08580         B
449   27.22    21.87     0.10940      0.19140    0.28710         M
280   11.62    18.18     0.11750      0.14830    0.10200         B
336   14.74    25.42     0.08275      0.07214    0.04105         B
436   12.77    22.47     0.09055      0.05761    0.04711         M
447   15.46    11.89     0.12570      0.15550    0.20320         M
367   16.02    23.24     0.08206      0.06669    0.03299         M
72    12.25    17.94     0.08654      0.06679    0.03885         B
301   13.16    20.54     0.07335      0.05275    0.01800         B
'''

#We will use seaborn to create plots 
import seaborn as sns

#Matplotlib will help us to draw the plots
import matplotlib.pyplot as plt
sns.set(color_codes=True)

#Let's extract the output variable using it's column name 
y = data.diagnosis

#Here we will plot it, and count instances for different class
ax = sns.countplot(y,label="Count")       
B, M = y.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)

#Here show the created plots 
plt.show()

#We will plot the violin plot for 10 features
#in which we don't want id and output variable
#So we will drop the id and diagnosis column from our data

list = ['id','diagnosis','diagnosis_numeric']
x = data.drop(list,axis = 1 )

#Let's print first 10 features after removing the selected columns
print(x.columns[0:10])

'''
Index(['diagnosis_numeric', 'radius', 'texture', 'perimeter', 'area',
       'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry'],
      dtype='object')
'''

#Output variable
data_dia = y

#Input variable
data = x

#Normalization of data
data_n_2 = (data - data.mean()) / (data.std())

#Select 10 features to visualize
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)

#Convert data into a form where, features can be separated on the basis
#of classes
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')

'''
    diagnosis              features     value
0            B             radius -0.685942
1            M             radius  0.727206
2            M             radius  0.968405
5660         B  fractal_dimension  0.483317
5661         M  fractal_dimension  2.583775
5662         M  fractal_dimension -0.782907
5663         M  fractal_dimension -1.095922
'''

#Figure dimensions
plt.figure(figsize=(7,7))

#Define Axes names
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")

#Rotate x axis numbers
plt.xticks(rotation=90)

#Show violin plot
plt.show()

#Joint Plot for correlation
sns.jointplot(x.loc[:,'concavity_worst'], 
              x.loc[:,'concave_points_worst'], 
              kind="regg", color="#ce1414")
plt.show()

#Let's plot correlation using grid plot

#We will use white background
sns.set(style="white")

#We will choose three features for this task
df = x.loc[:,['radius','compactness','texture']]

#Let's plot pairplot for correlation between above three 
sns.pairplot(df)

#Let's plot altogether
plt.show()

#Now let's plot correlation between all the features

#Define figure size
f,ax = plt.subplots(figsize=(15, 15))

#Create correlation plot using seaborn
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
corr = x.corr()

#plot the correlations
plt.show()
    
drop_list1 = ['perimeter','radius','compactness','concave_points',\
              'radius_se','perimeter_se','radius_worse','perimeter_worst',\
              'compactness_worst','concave_points_worst','compactness_se',\
              'concave_points_se','texture_worst','area_worst']

#Let's remove the redundant features
data = x.drop(drop_list1,axis = 1) 
print(data.columns)

'''
Index(['texture', 'area', 'smoothness', 'concavity', 'symmetry',
       'fractal_dimension', 'texture_se', 'area_se', 'smoothness_se',
       'concavity_se', 'symmetry_se', 'fractal_dimension_se',
       'smoothness_worst', 'concavity_worst', 'symmetry_worst',
       'fractal_dimension_worst'],
      dtype='object')
'''