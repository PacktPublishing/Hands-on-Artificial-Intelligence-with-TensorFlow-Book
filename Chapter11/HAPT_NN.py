'''
Created on 23-Jun-2018

@author: Ankit Dixit
'''

'''
Created on 22-Jun-2018

@author: Ankit Dixit
'''
#Tensorflow to build the classifier 
import tensorflow as tf
import keras

#Numpy for algebric operations as well as, 
#for loading and storing text files
import numpy as np

#Load training data and labels from disk using numpy
X_tr = np.loadtxt('E:/HAPT Data Set/Train/X_train.txt',
                  delimiter=' ').astype(np.float64)
Y_tr = np.loadtxt('E:/HAPT Data Set/Train/y_train.txt').astype(np.int32)

#Load test data and labels from disk using numpy
X_test = np.loadtxt('E:/HAPT Data Set/Test/X_test.txt',
                    delimiter=' ').astype(np.float64)
Y_test = np.loadtxt('E:/HAPT Data Set/Test/y_test.txt').astype(np.int32)

#We will start by creating place holders for
#Input as well as output variables
x = tf.placeholder('float', [None, 561])
y = tf.placeholder('float')

#As we will be using dropout in our network
#We need to create one placeholder for that 
dropout = tf.placeholder(tf.float32,[])

#Use Keras utility to convert 
#output variables into one hot encode
Y_tr = keras.utils.to_categorical(Y_tr-1)
Y_test = keras.utils.to_categorical(Y_test-1)

#Neural Network
def neural_net(X):
    
    #X: Input data set
    
    #Let's first define number of neurons in hidden layers
    nhidden = 1024
    
    #We need to create weight and bias variable 
    #in form of tensorflow tensors
    #for first hidden layer, there will be a 
    #weight matrix with 561X1024 dimension
    #we will initialize network weights and bias from normal distribution 
    W1 = tf.Variable(0.001*np.random.randn(561,nhidden).astype(np.float32),
                      name='weights')
    b1 = tf.Variable(0.001*np.random.randn(nhidden).astype(np.float32),
                      name='bias')
    
    #For second hidden layer number of neuron will be same as previous
    #weights in this layer will also be initialized from normal distribution
    W2 = tf.Variable(0.001*np.random.randn(nhidden,12).astype(np.float32),
                      name='weights')
    b2 = tf.Variable(0.001*np.random.randn(12).astype(np.float32),
                      name='bias')
    
    #So this is our first hidden layer with W1 and B1
    #we will use relu as activation with dropout   
    h0 = tf.nn.dropout(tf.nn.relu(tf.matmul(X,W1) + b1), dropout)
    
    #Finally second hidden layer which will generate predictions    
    pred = tf.matmul(h0,W2)+b2
    
    #Return predictions 
    return pred

def train_neural_network(x):
    
    #X: input data set
    
    #Let's create and initialize our neural network first
    prediction = neural_net(x)
    
    #Here we will define our cost function
    #We will be using cross entropy for calculating the loss
    #between actual and predicted values
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                  logits=prediction)
    cost = tf.reduce_mean(loss)
    
    #Let's call our optimizer
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    #Create a tensorflow session here
    sess = tf.InteractiveSession()
    
    #Before starting training of our network
    #we need to initialize all the variables
    #created so far.
    tf.global_variables_initializer().run()
    
    #We will run training loop for 1000 epochs
    for epoch in range(1000):
        loss = 0
        
        #Let's start cost function optimization
        #We will use 50% dropout probability
        _, c = sess.run([optimizer, cost], 
                        feed_dict = {x: X_tr, y: Y_tr, dropout:0.5})
        loss += c
        
        #Print training status after 100 epochs
        if (epoch % 100 == 0 and epoch != 0):
            print('Epoch', epoch, 'completed out of', 1000, 
                  'Training loss:', loss)
    
    #Once training completed we need to check performance
    #of trained network
    
    #Let's see how much instances our network successfully classified 
    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    
    #On the basis of number of correct prediction we can get accuracy
    #of our network
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), 
                              name='op_accuracy')
    
    #Let's run accuracy check for training and testing data set
    print('Train set Accuracy:', sess.run(accuracy, feed_dict = 
                                          {x: X_tr, y: Y_tr,dropout:1}))
    print('Test set Accuracy:', sess.run(accuracy, feed_dict = 
                                         {x: X_test, y: Y_test,dropout:1}))
    
#Let's start training of our classifier
train_neural_network(x)

'''
Epoch 100 completed out of 1000 Training loss: 0.06869250535964966
Epoch 200 completed out of 1000 Training loss: 0.026242759078741074
Epoch 300 completed out of 1000 Training loss: 0.014924143441021442
Epoch 400 completed out of 1000 Training loss: 0.009997222572565079
Epoch 500 completed out of 1000 Training loss: 0.007330700289458036
Epoch 600 completed out of 1000 Training loss: 0.005800022277981043
Epoch 700 completed out of 1000 Training loss: 0.0041691516526043415
Epoch 800 completed out of 1000 Training loss: 0.003301241435110569
Epoch 900 completed out of 1000 Training loss: 0.0028064025100320578
Train set Accuracy: 0.999485
Test set Accuracy: 0.945604

'''