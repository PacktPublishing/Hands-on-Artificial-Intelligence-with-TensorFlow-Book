'''
Created on 12-Jul-2018

@author: Ankit Dixit
'''

#Let's start with downloading and load our data set in matrix form

#Import pandas to manage data set 
import pandas as pd

#Import Tensorflow to get the file from given URL 
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
#Import NumPy for all mathematics operations on numerical data
import numpy as np

#Here's the name of our data file
FILE_PATH = 'bank_data_feat_select.csv'

#Let's read it using pandas and create a data frame
data = pd.read_csv(FILE_PATH)                                           

#Let's keep a copy of our data set
data_train_test = data.copy()

# The data needs to be split into a training set and a test set
# We will use 80% data for training and 20% for testing the trained model
training_instances = .8

#Let's divide Train and Test data into two matrices
total_instances = len(data)
training_set_size = int(total_instances * training_instances)
test_set_size = total_instances - training_set_size
print('Training Instances: %d\nTest Instances: %d'%(training_set_size,test_set_size))

labels = ['y']
features = data.columns
features = features[1:-2]
data = data[features]

#Data normalization
data = (data - data.mean()) / (data.std())

#Training Set
training_features = data.head(training_set_size)[features].copy()
training_labels = data_train_test.head(training_set_size)[labels].copy()

#Test Set
testing_features = data.tail(test_set_size)[features].copy()
testing_labels =   data_train_test.tail(test_set_size)[labels].copy()

#So data set part is done, now we can go ahead to configure our classifier

#We will create a DNN classifier using Tensorflow
#With 3 hidden layers of following size
hidden_units_spec = [200,500,200]

#Here we will define number of output classes 
n_classes_spec = 2
 
#Let's define number of training steps
steps_spec = 1000

#The number of epochs
epochs_spec = 15000

#We will store model and checkpoints in a temporary folder
#remember that whenever you do a new training session delete
#temporary folder, so it will not create any conflicts 
model_path = "tmp/model"

#As we are training a DNNClassifier
#It requires to define columns names before creating the classifier

#So we will do it with the help of our feature's list we created earlier
feature_columns = [tf.feature_column.numeric_column(key) 
                   for key in features]

#Here we will define our classifier
print('Creating Classifier...')
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, 
                                        hidden_units=hidden_units_spec, 
                                        activation_fn=tf.nn.relu,
                                        dropout = 0.5,
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

#Here's the name of our data file
FILE_PATH_Test = 'bank_data_feat_select_test.csv'

#Let's read it using pandas and create a data frame
data_test = pd.read_csv(FILE_PATH_Test)         

#Test Set
testing_features = data_test[features].copy()
testing_features =  (testing_features - data.mean()) / (data.std())
testing_labels =   data_test[labels].copy()                                  

print('Testing Classifier...')
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=testing_features, 
                                                    y=testing_labels,                                                     
                                                    shuffle=False)

#Let's evaluate the classifier on test data 
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("Accuracy on test set = {}".format(100*accuracy_score))

#For evaluation of our classifier we will use confusion matrix
from sklearn.metrics import confusion_matrix

#Import seaborn to create plots
import seaborn as sns

#Matplotlib to show the plots
import matplotlib.pyplot as plt

#Get predictions 
predictions = list(classifier.predict(input_fn=test_input_fn))

#Convert predictions in readable format
predictions = [p["class_ids"] for p in predictions] 

#Create confusion matrix
cm = confusion_matrix(testing_labels,predictions)

#Create Heat map of confusion matrix
sns.heatmap(cm,annot=True,fmt="d")

#Show the heatmap of confusion matrix
plt.show()
