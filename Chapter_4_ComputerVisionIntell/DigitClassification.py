'''
Created on 04-Oct-2018

@author: aii32199
'''

#Import necessary packages for building our CNN
from keras.models import Sequential

#We will Need convolutional layer for feature maps and max pooling layers
#Flatten layer will convert 2D image into 1D array for last layer computations
from keras.layers import Conv2D,MaxPool2D,\
Activation,Flatten,Dense

# Imports for array-handling and plotting
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.utils import np_utils 
import matplotlib.pyplot as plt
import numpy as np

#Here is our Network definition
def LeNet(width, height, depth, classes, weightsPath=None):
    
    #Initialize model
    model = Sequential()
    
    #Convolution => Activation(ReLu)=> pooling(Max Pooling)
    model.add(Conv2D(20,(5,5),padding='same',
                     input_shape=(depth,height,width)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    #Convolution => Activation(ReLu) => pooling(Max Pooling)
    model.add(Conv2D(50,(5,5),padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    #Fully connected Layer FC ==> ReLu
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    
    # If a pre-trained model is supplied
    if weightsPath is not None:
        model.load_weights(weightsPath)
    
    #return the constructed model
    return model

# Keras imports for the data set and building our neural network
#Let's Start by loading our data set
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# Print the shape before we reshape and normalize
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)
 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
 
# Normalizing the data to between 0 and 1 to help with the training
X_train /= 255
X_test /= 255
 
# Print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)
  
#One-hot encoding using keras' numpy-related utilities
n_classes = 10

# print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# Now let's define path to store the model
path2save = 'E:/PyDevWorkSpaceTest/Ensembles/Chapter_10/keras_mnist_lenet.h5'

#Build the model structure  
model = LeNet(28, 28, 1, 10)

#We will only store the best model with highest validation accuracy 
modelCheck = ModelCheckpoint(path2save, monitor='val_acc', 
                                       verbose=0, save_best_only=True, 
                                       save_weights_only=True, mode='auto')

#Optimizer will be adaptive momentum with categorical loss
model.compile(optimizer="Adam", 
              loss = "categorical_crossentropy",metrics=["accuracy"])
 
# Start training the model and saving metrics in history
history = model.fit(X_train, Y_train,
          batch_size=128, epochs=20,
          verbose=2,
          validation_data=(X_test, Y_test),
          callbacks= [modelCheck])
  
print('Saved trained model at %s ' % path2save)

# Plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
  
plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
plt.show()


mnist_model = LeNet(28, 28, 1, 10, path2save)

#We will use Evaluate function
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)
  
print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])

#Load the model and create predictions on the test set
predicted_classes = mnist_model.predict_classes(X_test)
  
#See which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

#Adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7,14)
  
plt.figure()
  
# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), 
               cmap='gray', interpolation='none')
    plt.title("Predicted: {}, Truth: {}".format(
                       predicted_classes[correct],
                        y_test[correct]))
    plt.xticks([])
    plt.yticks([])
  
  
# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(X_test[incorrect].reshape(28,28), 
               cmap='gray', interpolation='none')
    plt.title("Predicted {}, Truth: {}".format(
                    predicted_classes[incorrect], 
                    y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])
  
plt.show()
