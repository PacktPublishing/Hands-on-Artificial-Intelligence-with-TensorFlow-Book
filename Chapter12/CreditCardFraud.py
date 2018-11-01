'''
Created on 14-Aug-2018

@author: Ankit Dixit
'''
# We will call Keras callbacks to monitor classifier
from keras.callbacks import ModelCheckpoint, TensorBoard

# Import layers to construct the neural network
from keras.layers import Dense, Dropout,Layer

# Network type
from keras.models import Sequential

# To fix the size of all figures
from pylab import rcParams

# We will use sklearn to create training and testing data
# and for normalization of the data set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# matplotlib to create different plots
import matplotlib.pyplot as plt

# Pandas for data set related operation
import pandas as pd

# Seaborn to create distribution plots
import seaborn as sns

# Numpy for matrix based operations
import numpy as np

# Let's initialize some parameters
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

# Let's load the data set into the memory
df = pd.read_csv("creditcard.csv")
print(df.shape)
'''(284807, 31)'''

# Let's check the class distribution first
count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency");
# plt.show()

frauds = df[df.Class == 1]
normal = df[df.Class == 0]
print(frauds.shape)
'''(492, 31)'''
print(normal.shape)
'''(284315, 31)'''

# Print stats for fraud cases
print(frauds.Amount.describe())

'''
count     492.000000
mean      122.211321
std       256.683288
min         0.000000
25%         1.000000
50%         9.250000
75%       105.890000
max      2125.870000
Name: Amount, dtype: float64
'''

print(normal.Amount.describe())

'''
count    284315.000000
mean         88.291022
std         250.105092
min           0.000000
25%           5.650000
50%          22.000000
75%          77.050000
max       25691.160000
Name: Amount, dtype: float64
'''

# Let's plot histogram for fraudulent and real transactions
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')

bins = 50
ax1.hist(frauds.Amount, bins = bins)
ax1.set_title('Fraud')

ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
# plt.show()

# Drop time column from the data as it is not much useful
data = df.drop(['Time'], axis=1)

# Here we will normalize the data using Standard scaler from sklearn
data['Amount'] = \
StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# Let's create training and test set from the data set
# we will choose 80% data for training and 20% for testing
X_train, X_test = train_test_split(data, 
                                   test_size=0.2, 
                                   random_state=RANDOM_SEED)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)
y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)
X_train = X_train.values
X_test = X_test.values
print(X_train.shape)
'''(227451, 29)'''

# Following method create our autoencoder
def AutoEncoder(input_dim,encoding_dim,drop,weights=None):
    
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
    net.add(Dense(int(encoding_dim / 2), activation="relu"))
    net.add(Dropout(drop))
    # Third hidden layer
    net.add(Dense(int(encoding_dim / 2), activation='relu'))
    # Output layer
    net.add(Dense(input_dim, activation='linear'))
    
    # If trained weights provided load into the network
    if weights is not None:
        net.load_weights(weights)
    
    # Return the network
    return net

# Define hyper-parameters for the auroencoder

#Input dimension
input_dim = X_train.shape[1]
# Number of neuron in first hidden layer
encoding_dim = 28
# We will use 10% dropout
drop = 0.10
# We will train the classifier for 100 epochs
nb_epoch = 200
# with 256 samples in a batch
batch_size = 256 #32
# Optimizer
opt = 'adam'
# Cost function to minimize
loss_func = 'mean_squared_error'
# Model name
model_name = "autoencoder_weights.h5"

# Let's create the autoencoder here
autoencoder = AutoEncoder(input_dim, encoding_dim, drop)

# We will compile the model by defined optimizer and loss function
autoencoder.compile(optimizer=opt, 
                    loss=loss_func, 
                    metrics=['accuracy'])

# Model check point will help us in monitoring the autoencoder'training
checkpointer = ModelCheckpoint(filepath=model_name,
                               verbose=0,
                               save_best_only=True,
                               save_weights_only=True)

# Let's train our AE here
history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
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
ax.set_title('Autoencoder Training Summary')
plt.legend(['train', 'test'], loc='upper right');
plt.show()

# This is the time to test the classifier 
autoencoder = AutoEncoder(input_dim, encoding_dim, 
                          drop,'autoencoder_weights.h5')

predictions = autoencoder.predict(X_test)

# Calculate reconstruction error between input and output
mse = np.mean(np.power(X_test - predictions, 2), axis=1)

# Let's see the behaviour of the reconstruction error
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
print(error_df.describe())

'''
        reconstruction_error    true_class
count          56962.000000  56962.000000
mean               0.307041      0.001720
std                1.375394      0.041443
min                0.024482      0.000000
25%                0.130933      0.000000
50%                0.198664      0.000000
75%                0.304401      0.000000
max               93.491394      1.000000
'''

# Error for the normal classes 
fig,(ax1,ax2) = plt.subplots(2,1)
normal_error_df = error_df[(error_df['true_class']== 0)]
ax1.set_title('Reconstruction error for Real and Fraud Transactions')
ax1.set_ylabel('Number of Transaction')
_ = ax1.hist(normal_error_df.reconstruction_error.values, bins=10)

# Error for the fraud class
fraud_error_df = error_df[error_df['true_class'] == 1]
ax2.set_ylabel('Number of Transaction')
ax2.set_xlabel('Error Bins')
_ = ax2.hist(fraud_error_df.reconstruction_error.values, bins=10)

plt.show()
# Let's see confusion matrix it will help us 
# to understand performance of our classifier
from sklearn.metrics import confusion_matrix 
          
# We need to define a threshold value
# which will classify the input value into
# one of the class                             
threshold = 2.5

# Let's visualize our decision boundary to see
# how well it is discriminate the data 
groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();

# Here we will create confusion matrix
# It will help us to get information related to 
# false positives
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()                             