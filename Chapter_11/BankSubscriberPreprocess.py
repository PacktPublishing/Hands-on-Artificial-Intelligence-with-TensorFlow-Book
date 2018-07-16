'''
Created on 09-Jul-2018

@author: Ankit Dixit
'''

# We will use pandas for creating data frames
from sklearn.preprocessing import LabelEncoder

import pandas as pd


# scikit learn will help us in normalizing our data set 
# Let's start with loading our data from csv file
data = pd.read_csv('bank-additional.csv', sep = ";")

#Let's verify the size of data set
print('Number of Instances: %d\nNumber of attributes: %d'%(data.shape[0],data.shape[1]))

'''
Number of Instances: 41188
Number of attributes: 21
'''

# Load attributes in a python list 
var_names = data.columns.tolist()
print(var_names)

'''

['age', 'job', 'marital', 'education', 'default', 'housing', 
'loan', 'contact', 'month', 'day_of_week', 'duration', 
'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 
'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']

'''

# We have already identified the categorical variable
cat_var = ['job','marital','education','default',
          'housing','loan','contact','month',
          'day_of_week','poutcome','y']

# Let's see the variable values
print(data.head(5)[cat_var[:7]]) 

'''
         job  marital    education  default housing loan    contact
0  housemaid  married     basic.4y       no      no   no  telephone
1   services  married  high.school  unknown      no   no  telephone
2   services  married  high.school       no     yes   no  telephone
3     admin.  married     basic.6y       no      no   no  telephone
4   services  married  high.school       no      no  yes  telephone
'''

# Let's create a separate list for numeric attributes
numeric_var = [i for i in var_names if i not in cat_var]

# Let's create a data frame with numeric variables only
df1 = data[numeric_var]
df1_names = df1.keys().tolist()

# Assign attribute name to the data frame
df1.columns = df1_names

# Now we will handle categorical values
# sklear's LabelEncoder will help us to create numeric mapping 
# of our categorical variables

# We will create a new data frame to store the numerical values 
encode = LabelEncoder()
data_1 = pd.DataFrame(index=range(data.shape[0]),columns=cat_var) 
data_1['job'] = encode.fit_transform(data['job'])
data_1['marital'] = encode.fit_transform(data['marital'])
data_1['education'] = encode.fit_transform(data['education'])
data_1['default'] = encode.fit_transform(data['default'])
data_1['housing'] = encode.fit_transform(data['housing'])
data_1['loan'] = encode.fit_transform(data['loan'])
data_1['contact'] = encode.fit_transform(data['contact'])
data_1['day_of_week'] = encode.fit_transform(data['day_of_week'])
data_1['poutcome'] = encode.fit_transform(data['poutcome'])

print(data['month'].unique())
'''
['may' 'jun' 'jul' 'aug' 'oct' 'nov' 'dec' 'mar' 'apr' 'sep']
'''

data_1 = data_1.drop(['month'],axis=1)
import numpy as np
# Get all the values from data frame
data_lst = [data]

# Create a attribute with the numeric values of the months.
for attribute in data_lst:
    attribute.loc[attribute["month"] == "jan", "month_integer"] = 1
    attribute.loc[attribute["month"] == "feb", "month_integer"] = 2
    attribute.loc[attribute["month"] == "mar", "month_integer"] = 3
    attribute.loc[attribute["month"] == "apr", "month_integer"] = 4
    attribute.loc[attribute["month"] == "may", "month_integer"] = 5
    attribute.loc[attribute["month"] == "jun", "month_integer"] = 6
    attribute.loc[attribute["month"] == "jul", "month_integer"] = 7
    attribute.loc[attribute["month"] == "aug", "month_integer"] = 8
    attribute.loc[attribute["month"] == "sep", "month_integer"] = 9
    attribute.loc[attribute["month"] == "oct", "month_integer"] = 10
    attribute.loc[attribute["month"] == "nov", "month_integer"] = 11
    attribute.loc[attribute["month"] == "dec", "month_integer"] = 12

# Change datatype from int32 to int64
data_1["month_integer"] = data["month_integer"].astype(np.int64)
print(data_1["month_integer"].unique())

'''
[ 5  6  7  8 10 11 12  3  4  9]
'''

# As our output variable is a binary we will choose
# 0 for NO and 1 for YES; we will do it with the help 
# of dictionary
dict_map = dict()
y_map = {'yes':1,'no':0}
dict_map['y'] = y_map
data = data.replace(dict_map)
label = data['y']
data_1['y'] = label

# To create final data frame join both the data frames 
final_df = pd.concat([df1,data_1],axis=1)    

#Let's verify the size of data set
print('Number of Instances: %d\nNumber of attributes: %d'%(final_df.shape[0],final_df.shape[1]))

'''
Number of Instances: 41188
Number of attributes: 21
'''

# Let's see pre-processed data frame
print(final_df.head(5)[cat_var[:5]])

'''
     job  marital  education  default  housing
0    3        1          0        0        0
1    7        1          3        1        0
2    7        1          3        0        2
3    0        1          1        0        0
4    7        1          3        0        0
'''

# Store data frame to the disk
final_df.to_csv('bank_data_test.csv', index = False)