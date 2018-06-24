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

#File Name - be sure to change this if you upload something else
file_name = "wdbc.csv"

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

# The data needs to be split into a training set and a test set
# We will use 80% data for training and 20% for testing the trained model
training_instances = .8

#Let's divide Train and Test data into two matrices
total_instances = len(data)
training_set_size = int(total_instances * training_instances)
test_set_size = total_instances - training_set_size
print('Training Instances: %d\nTest Instances: %d'%(training_set_size,test_set_size))

'''
Training Instances: 455
Test Instances: 114
'''

#Now lets divide our data set in train and test set

#Before that define our output variable 
labels = ['diagnosis_numeric']

#We will use all the features for training purpose
features = ['radius', 'texture',
       'perimeter', 'area', 'smoothness', 'compactness', 'concavity',
       'concave_points', 'symmetry', 'fractal_dimension', 'radius_se',
       'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worse', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave_points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

#Training Set
training_features = data.head(training_set_size)[features].copy()
training_labels = data.head(training_set_size)[labels].copy()

#Test Set
testing_features = data.tail(test_set_size)[features].copy()
testing_labels = data.tail(test_set_size)[labels].copy()

#So data set part is done, now we can go ahead to configure our classifier

#We will create a DNN classifier using Tensorflow
#With 3 hidden layers of following size
hidden_units_spec = [10,20,10]

#Here we will define number of output classes 
n_classes_spec = 2
 
#Let's define number of training steps
steps_spec = 2000

#The number of epochs
epochs_spec = 25

#We will store model and checkpoints in a temporary folder
#remember that whenever you do a new training session delete
#temporary folder, so it will not create any conflicts 
model_path = "tmp/model"

#As we are training a DNNClassifier
#It requires to define columns names before creating the classifier

#So we will do it with the help of our feature's list we created earlier
feature_columns = [tf.feature_column.numeric_column(key) for key in features]

#Here we will define our classifier
print('Creating Classifier...')
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, 
                                        hidden_units=hidden_units_spec, 
                                        n_classes=n_classes_spec, 
                                        model_dir=model_path)

#Now as we have our training and testing data in form of 
#pandas data frame, we need to convert it into tensorflow tensor,
#tensorflow input function will help us to do the same
print('Creating Inputs...')
train_input_fn = tf.estimator.inputs.pandas_input_fn(x=training_features, 
                                                     y=training_labels, 
                                                     num_epochs=epochs_spec, 
                                                     shuffle=True)

#Let's Train the model using the classifier.
print('Classifier Training...')
classifier.train(input_fn=train_input_fn, steps=steps_spec)


#We need to convert test data frame also in the tensor form
#we will do the same with using tensorflow input function
print('Testing Classifier...')
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=testing_features, 
                                                    y=testing_labels, 
                                                    num_epochs=epochs_spec, 
                                                    shuffle=False)

#Let's evaluate the classifier on test data 
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("Accuracy on test set = {}".format(accuracy_score))