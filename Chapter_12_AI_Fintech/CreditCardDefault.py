'''
Created on 16-Aug-2018

@author: Ankit Dixit
'''
# Let's Start with importing all required packages
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Layer
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split

# wget will help us to download data from the repository
import wget


import numpy as np

import seaborn as sns

# Numpy for matrix related operations
# Let's initialize random number generation seed
np.random.seed(1337)

# pandas for data set related operations
import pandas as pd

# matplolib for visualization of plots
import matplotlib.pyplot as plt

# For directory related operations import OS
import os

# Let's initialize parameters for plots
plt.rcParams['figure.figsize'] = (12.0, 12.0)
plt.rcParams.update({'font.size': 10})
plt.rcParams['xtick.major.pad']='5'
plt.rcParams['ytick.major.pad']='5'
plt.style.use('ggplot')

# Define directory to store data set
datadir = './data'

# If directory is not available create it
if not os.path.exists(datadir):
    os.makedirs(datadir)

# Get the dataset from UCI repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases\
/00350/default of credit card clients.xls' 

# Get the full path  to store the data
filename = os.path.join(datadir, 'default of credit card clients.xls')

# Download the data if not on the disk
if not os.path.isfile(filename):
    wget.download(url, out=filename)

# Read downloaded data into pandas dataframe 
df = pd.read_excel(filename, header=1)

# Here we will rename the column names 
df.columns = [x.lower() for x in df.columns]
df = df.rename(index=str, columns={"pay_0": "pay_1"})

# Drop customer ID 
df = df.drop('id', axis=1)
# sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f')
# plt.show()
print(df.columns)

# Let's see number of the variables
print("Explanatory variables:  {}".format(len(df.columns)-1))
print("Number of Observations: {}".format(df.shape[0]))

# Get the class variable
df['target'] = df['default payment next month']
df = df.drop(['default payment next month'],axis=1)

# Let's check the class distribution first
LABELS = ['Normal','Default']
count_classes = pd.value_counts(df['target'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Number of Instances");
# plt.show()

# Now let's see how demographic information affects
# output variable.
# We will choose 'Sex', 'Marriage' and 'Age' information for this
df_copy = df.copy()
df_copy['sex'] = df_copy['sex'].astype('category').cat.rename_categories(['M', 'F'])
df_copy['marriage'] = df_copy['marriage'].astype('category').cat.rename_categories(['na', 'married', 'single', 'other'])
df_copy['age_cat'] = pd.cut(df_copy['age'], range(0, 100, 10), right=False)
df_copy['target'] = df_copy['target'].astype('category')
fig, ax = plt.subplots(1,3)
fig.set_size_inches(20,5)
fig.suptitle('Defaulting by absolute numbers, for various demographics')

d = df_copy.groupby(['target', 'sex']).size()
p = d.unstack(level=1).plot(kind='bar', ax=ax[0])
d = df_copy.groupby(['target', 'marriage']).size()
p = d.unstack(level=1).plot(kind='bar', ax=ax[1])
d = df_copy.groupby(['target', 'age_cat']).size()
p = d.unstack(level=1).plot(kind='bar', ax=ax[2])

# plt.show()
columns_to_drop = ['bill_amt1','bill_amt2',
                   'bill_amt3','bill_amt4',
                   'bill_amt5','bill_amt6']

df = df.drop(columns_to_drop,axis=1)

X_train, X_test,Y_train,Y_test = train_test_split(df,df['target'], test_size=0.1,random_state=212)

X_train = X_train.drop(['target'],axis=1)
X_test = X_test.drop(['target'],axis=1)

X_train = (X_train-X_train.mean())/X_train.std()
X_test =  (X_test-X_train.mean())/X_train.std()

X_train = X_train.values
X_test = X_test.values

Y_train = Y_train.values#to_categorical(Y_train.values, 2)
Y_test = Y_test.values#to_categorical(Y_test.values, 2)
 
print('Training Set Shape: ',X_train.shape)
print('Test Set Shape: ',X_test.shape)


# Following method create our net
def NeuralNetwork(input_dim,encoding_dim,drop,weights=None):
    
    #INPUT_DIM: is input size
    #ENCODING_DIM: is hidden layer size
    #DROP: Droupout for regularization 
    #WEIGHTS: If pre-trained weights available
    
    # We will create a sequential network
    net = Sequential()
    # Our input layer
    net.add(Layer(input_shape=(input_dim, )))
    # Lets add first hidden layer with relu activation
    net.add(Dense(encoding_dim, activation="relu"))
    # Add some dropout for regularization
    net.add(Dropout(drop))               
    # Second hidden layer 
    net.add(Dense(int(encoding_dim), activation="relu"))
    net.add(Dropout(drop))
    # Third hidden layer
    net.add(Dense(int(encoding_dim / 2), activation='relu'))
    net.add(Dropout(drop))
    # Fourth hidden layer
    net.add(Dense(int(encoding_dim / 2), activation='relu'))
    # Output layer
    net.add(Dense(1, activation='sigmoid'))
    
    # If trained weights provided; load into the network
    if weights is not None:
        net.load_weights(weights)
    
    # Return the network
    return net

# Define hyper-parameters for the autoencoder

#Input dimension
input_dim = X_train.shape[1]
# Number of neuron in first hidden layer
encoding_dim = 512
# We will use 10% dropout
drop = 0.10
# We will train the classifier for 100 epochs
nb_epoch = 1000
# with 256 samples in a batch
batch_size = 32 #32
# Optimizer
opt = 'adam'
# Cost function to minimize
loss_func = 'mse'
# Model name
model_name = "network_weights.h5"

# Let's create the net here
net = NeuralNetwork(input_dim, encoding_dim, drop)

# We will compile the model by defined optimizer and loss function
net.compile(optimizer=opt, 
                    loss=loss_func, 
                    metrics=['accuracy'])

# Model check point will help us in monitoring the net training
checkpointer = ModelCheckpoint(filepath=model_name,                               
                               monitor='loss',
                               save_best_only=True,
                               save_weights_only=True,
                               mode='auto')

# Let's train our AE here
history = net.fit(X_train, Y_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,                    
                    validation_data=(X_test, Y_test),
                    verbose=2,
                    callbacks=[checkpointer]).history
   
# We will plot training summary in following lines
loss,val_loss = history['loss'],history['val_loss']
epochs = range(0,len(loss))
ax = plt.axes()
ax.set_xlabel('Epochs',fontsize=12)
ax.set_ylabel('Loss',fontsize=12)
ax.plot(epochs,loss)
ax.plot(epochs,val_loss)
ax.set_title('Network Training Summary')
plt.legend(['train', 'test'], loc='upper right');
# plt.show()

test_net = NeuralNetwork(input_dim, encoding_dim, drop, weights='network_weights.h5')
prediction = test_net.predict(X_test)

predictions = []

for pred in prediction:
    if pred>0.5:
        pred = 1
    else:
        pred = 0
    predictions.append(pred)

predictions = np.array(predictions)


conf_matrix = confusion_matrix(Y_test, predictions)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()              

