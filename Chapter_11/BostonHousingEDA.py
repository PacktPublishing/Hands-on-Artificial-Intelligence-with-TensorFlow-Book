'''
Created on 13-Jul-2018

@author: Ankit Dixit
'''

# We will import data set from sklearn
from sklearn.datasets import load_boston

# Load data set 
data = load_boston()
 
# Print dictionary keys 
print(data.keys())

'''
dict_keys(['data', 'target', 'DESCR', 'feature_names'])
'''

# Print feature description
#print(data.DESCR)

# import Pandas for data frame operations
import pandas as pd

# Create Pandas data frame using dictionary fields
# data contains data and feature_names will be columns 
data_frame = pd.DataFrame(data.data,columns=data.feature_names)

# Let's verify the size of data set
print('Number of Instances: %d\nNumber of attributes: %d'%(data_frame.shape[0],data_frame.shape[1]))

print(data_frame.head(10)[data_frame.columns[:10]])

'''
     CRIM    ZN  INDUS  CHAS    NOX     RM    AGE     DIS  RAD    TAX
0  0.00632  18.0   2.31   0.0  0.538  6.575   65.2  4.0900  1.0  296.0
1  0.02731   0.0   7.07   0.0  0.469  6.421   78.9  4.9671  2.0  242.0
2  0.02729   0.0   7.07   0.0  0.469  7.185   61.1  4.9671  2.0  242.0
3  0.03237   0.0   2.18   0.0  0.458  6.998   45.8  6.0622  3.0  222.0
4  0.06905   0.0   2.18   0.0  0.458  7.147   54.2  6.0622  3.0  222.0
5  0.02985   0.0   2.18   0.0  0.458  6.430   58.7  6.0622  3.0  222.0
6  0.08829  12.5   7.87   0.0  0.524  6.012   66.6  5.5605  5.0  311.0
7  0.14455  12.5   7.87   0.0  0.524  6.172   96.1  5.9505  5.0  311.0
8  0.21124  12.5   7.87   0.0  0.524  5.631  100.0  6.0821  5.0  311.0
9  0.17004  12.5   7.87   0.0  0.524  6.004   85.9  6.5921  5.0  311.0
'''

# Create a copy of data frame
data_frame_target = data_frame.copy()

# Update new data frame by adding target  variable
data_frame_target['target'] = data.target 
print(data_frame_target.head(10)[data_frame_target.columns])  
 
'''
      CRIM    ZN  INDUS  CHAS   ...    PTRATIO       B  LSTAT  target
0  0.00632  18.0   2.31   0.0   ...       15.3  396.90   4.98    24.0
1  0.02731   0.0   7.07   0.0   ...       17.8  396.90   9.14    21.6
2  0.02729   0.0   7.07   0.0   ...       17.8  392.83   4.03    34.7
3  0.03237   0.0   2.18   0.0   ...       18.7  394.63   2.94    33.4
4  0.06905   0.0   2.18   0.0   ...       18.7  396.90   5.33    36.2
5  0.02985   0.0   2.18   0.0   ...       18.7  394.12   5.21    28.7
6  0.08829  12.5   7.87   0.0   ...       15.2  395.60  12.43    22.9
7  0.14455  12.5   7.87   0.0   ...       15.2  396.90  19.15    27.1
8  0.21124  12.5   7.87   0.0   ...       15.2  386.63  29.93    16.5
9  0.17004  12.5   7.87   0.0   ...       15.2  386.71  17.10    18.9

'''

# Stats for first 5 features
print('Feature Description for first five features')
print(data_frame.describe()[data_frame.columns[:5]])

# Stats for last 5 features
print('\nFeature Description for next five features')
print(data_frame.describe()[data_frame.columns[5:10]])

# Target Variable
print('\nFeature Description for target variable')
print(data_frame_target['target'].describe())

'''
Feature Description for first five features
             
             CRIM          ZN       INDUS        CHAS         NOX
count  506.000000  506.000000  506.000000  506.000000  506.000000
mean     3.593761   11.363636   11.136779    0.069170    0.554695
std      8.596783   23.322453    6.860353    0.253994    0.115878
min      0.006320    0.000000    0.460000    0.000000    0.385000
25%      0.082045    0.000000    5.190000    0.000000    0.449000
50%      0.256510    0.000000    9.690000    0.000000    0.538000
75%      3.647423   12.500000   18.100000    0.000000    0.624000
max     88.976200  100.000000   27.740000    1.000000    0.871000

Feature Description for next five features

               RM         AGE         DIS         RAD         TAX
count  506.000000  506.000000  506.000000  506.000000  506.000000
mean     6.284634   68.574901    3.795043    9.549407  408.237154
std      0.702617   28.148861    2.105710    8.707259  168.537116
min      3.561000    2.900000    1.129600    1.000000  187.000000
25%      5.885500   45.025000    2.100175    4.000000  279.000000
50%      6.208500   77.500000    3.207450    5.000000  330.000000
75%      6.623500   94.075000    5.188425   24.000000  666.000000
max      8.780000  100.000000   12.126500   24.000000  711.000000

Feature Description for target variable

count    506.000000
mean      22.532806
std        9.197104
min        5.000000
25%       17.025000
50%       21.200000
75%       25.000000
max       50.000000
Name: target, dtype: float64'''

# Import seaborn and matplolib for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Get prices from target variable
prices = data_frame_target['target']

# Create figure with properties
fig, axs = plt.subplots(nrows=3,ncols=1, figsize=(8,18))

# Plot distribution 
for i in range(len(axs)):
    axs[i].set_ylim([0, 60.0])
    
    # RM vs Price
_ = sns.regplot(x=data_frame['RM'], y=prices, ax=axs[0])
    
    # LSSAT vs Price
_ = sns.regplot(x=data_frame['LSTAT'], y=prices, ax=axs[1])
    
    # PTRATIO vs Price
_ = sns.regplot(x=data_frame['PTRATIO'], y=prices, ax=axs[2])

plt.show()

# Define figure size
f,ax = plt.subplots(figsize=(15, 15))

# Create correlation plot using seaborn
sns.heatmap(data_frame.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
corr = data_frame.corr()
plt.show()