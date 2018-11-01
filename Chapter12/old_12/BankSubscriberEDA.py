'''
Created on 10-Jul-2018

@author: Ankit Dixit
'''

# We will use seaborn to create plots 
import seaborn as sns

# Matplotlib will help us to draw the plots
import matplotlib.pyplot as plt
sns.set(color_codes=True)

# Import pandas to manage data set 
import pandas as pd

# Import NumPy for all mathematics operations on numerical data
import numpy as np

# Let's load the pre-processed version of data set 
file_name = 'bank_data_test.csv'

# Load into a variable using pandas read_csv
data = pd.read_csv(file_name, delimiter=',')

# Let's verify the size of data set
print('Number of Instances: %d\nNumber of attributes: %d'%(data.shape[0],data.shape[1]))

'''
Number of Instances: 41188
Number of attributes: 21
'''

# Let's see a brief summary of some variables 
print(data.describe()[['age','duration','campaign','pdays']])

'''
             age      duration      campaign         pdays
count  41188.00000  41188.000000  41188.000000  41188.000000
mean      40.02406    258.285010      2.567593    962.475454
std       10.42125    259.279249      2.770014    186.910907
min       17.00000      0.000000      1.000000      0.000000
25%       32.00000    102.000000      1.000000    999.000000
50%       38.00000    180.000000      2.000000    999.000000
75%       47.00000    319.000000      3.000000    999.000000
max       98.00000   4918.000000     56.000000    999.000000
'''

# Let's extract the output variable using it's column name 
y = data.y

# We will shuffle the data set before visualization
data = data.reindex(np.random.permutation(data.index))

# Here we will plot it, and count instances for different class
ax = sns.countplot(y,label="Count")       
No, Yes= y.value_counts()
print('Number of to be subscriber: ',Yes)
print('Number of not to be subscriber : ',No)

'''
Number of to be subscriber:  36548
Number of not to be subscriber :  4640
'''

# Here show the created plots 
plt.show()

# We will create 4 distribution plots
f, axes = plt.subplots(nrows=2,ncols=2, figsize=(15, 6))

# Monthly marketing activity
sns.distplot(data['month_integer'], kde=False, color="#ff3300", ax=axes[0][0]).set_title('Months of Marketing Activity Distribution')
axes[0][0].set_ylabel('Potential Clients Count')
axes[0][0].set_xlabel('Months')

# Potential subscriber on Age basis
sns.distplot(data['age'], kde=False, color="#3366ff", ax=axes[0][1]).set_title('Age of Potentical Clients Distribution')
axes[0][1].set_ylabel('Potential Clients Count')
axes[0][1].set_xlabel('Age')

# Potential subscriber on Job basis
sns.distplot(data['campaign'], kde=False, color="#546E7A", ax=axes[1][0]).set_title('Calls Received in the Marketing Campaign')
axes[1][0].set_ylabel('Potential Clients Count')
axes[1][0].set_xlabel('Campaign')

# Jobs
sns.distplot(data['job'], kde=False, color="#33ff66", ax=axes[1][1]).set_title('Potential clients on Job basis')
axes[1][1].set_ylabel('Potential Clients Count')
axes[1][1].set_xlabel('Job Type')

#Show all created plots
plt.show()

# We will first remove output variable from data
x = data

# Store output variable
y = data.y

# Now let's plot correlation between all the features

# Define figure size
f,ax = plt.subplots(figsize=(15, 15))

# Create correlation plot using seaborn
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
corr = x.corr()

# plot the correlations
plt.show()

# We will drop highly correlated features
drop_list = ['emp.var.rate','nr.employed','cons.price.idx','euribor3m','previous']

#Let's remove the redundant features
data = x.drop(drop_list,axis = 1) 
print(data.columns)

'''
Index(['age', 'duration', 'campaign', 'pdays', 'cons.conf.idx', 'job',
       'marital', 'education', 'default', 'housing', 'loan', 'contact',
       'day_of_week', 'poutcome', 'y', 'month_integer'],
      dtype='object')
'''

data.to_csv('bank_data_feat_select_test.csv')

