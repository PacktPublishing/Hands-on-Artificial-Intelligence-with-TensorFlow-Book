# Start with importing the libraries
# All of them now familiar to you
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

# Same data set is available with tensorflow too
# so we will use it same for this implementation
from tensorflow.contrib import learn

# Sklearn cross_validation will be used to create 
# Training and test set
from sklearn import cross_validation

# We will normalize our data set using sklearn 
from sklearn import preprocessing

# For RegressionModel evaluation we will use R2 score from sklearn
from sklearn import metrics

# Let's start with loading our data set
boston = learn.datasets.load_dataset('boston')

# Separate the input and output variables
x, y = boston.data, boston.target
y.resize( y.size, 1 ) 

# Create training and test set
train_x, test_x, train_y, test_y = cross_validation.train_test_split(
                                    x, y, test_size=0.2, random_state=42)

print( "Dimension of Boston test_x = ", test_x.shape )
print( "Dimension of test_y = ", test_y.shape )
print( "Dimension of Boston train_x = ", train_x.shape )
print( "Dimension of train_y = ", train_y.shape )

'''
Dimension of Boston test_x =  (102, 13)
Dimension of test_y =  (102, 1)
Dimension of Boston train_x =  (404, 13)
Dimension of train_y =  (404, 1)
number of features =  13
batch size =  50
test length=  404
number batches =  8
'''

#Here we will normalize the input
scaler = preprocessing.StandardScaler( )
train_x = scaler.fit_transform( train_x )
test_x  = scaler.fit_transform( test_x )

# Number of features
numFeatures =  train_x.shape[1] 
print( "number of features = ", numFeatures )

# Let's create name scope for input and output layer
# We will use tensorflow place holders
with tf.name_scope("IO"):
    inputs = tf.placeholder(tf.float32, [None, numFeatures], name="X")
    outputs = tf.placeholder(tf.float32, [None, 1], name="Yhat")

# Here we will create hidden layers for our network    
with tf.name_scope("LAYER"):
    
    # Here we will define layer architecture    
    Layers = [numFeatures, 128, 128, 1]
    
    #List to store Weights and bias for hidden layers
    W = []
    b = []
    
    #Let's initialize the weights and biases 
    for i in range( 1, len( Layers ) ):
        W.append( tf.Variable(tf.random_normal([Layers[i-1], 
                                                Layers[i]], 
                                               0, 0.1, 
                                               dtype=tf.float32), 
                              name="W%d" % i ) )
        b.append( tf.Variable(tf.random_normal([Layers[i]], 
                                               0, 0.1, 
                                               dtype=tf.float32 ), 
                              name="b%d" % i ) )
    
    #We will use High Dropout to reduce chances of over fitting 
    dropout = 0.990           
    keep_prob = tf.placeholder(tf.float32)   

def RegressionModel( inputs, W, b ):
    
    # Output from previous will be input of next layer
    lastY = inputs
    
    #Let's create layer operation W*Input+b
    for i, (Wi, bi) in enumerate( zip( W, b ) ):
        y =  tf.add( tf.matmul( lastY, W[i]), b[i] )    
        
        if i==len(W)-1:
            return y
        
        #As it is regression our last layer will have sigmoid activation
        lastY =  tf.nn.sigmoid( y )
        lastY =  tf.nn.dropout( lastY, dropout )

# We will create training configuration here
with tf.name_scope("train"):
        
    #Build model here with all weights and biases
    yout = RegressionModel( inputs, W, b )
    
    #Mean square error will be our cost function
    cost_op = tf.reduce_mean( tf.pow( yout - outputs, 2 ))
    
    #We will use SGD with Adaptive gradient 
    train_op = tf.train.AdamOptimizer().minimize( cost_op )


# Here are the hyper parameters

# Create a counter for counting epochs
epoch       = 0          

# Store cost from previous epoch 
last_cost   = 0          

# Maximum number of epochs
max_epochs  = 20000      

# Tolerance
tolerance   = 1e-6

# Batch Size       
batch_size  = 256        

# Following variables will tell us about number of batches.
num_samples = train_y.shape[0]                  
num_batches = int( num_samples / batch_size )   
    
print( "batch size = ", batch_size )
print( "test length= ", num_samples )
print( "number batches = ", num_batches )
print( "--- Beginning Training ---" )

'''batch size =  256
test length=  404
number batches =  1
--- Beginning Training ---'''

# Let's Create a tensorflow session
sess = tf.Session() 
with sess.as_default():
    
    # Initialize all variables
    init = tf.global_variables_initializer()
    
    # Initialize the session
    sess.run(init)
    
    # Start training until we stop, either because we've reached the max
    # number of epochs, or successive errors are close enough to each other
    # (less than tolerance)
    
    # Here we store cost at each epoch 
    costs = []
    epochs= []
    
    # Let's start training
    while True:
        
        cost = 0
        for n in range(  num_batches ):
            batch_x = train_x[ n*batch_size : (n+1)*batch_size ]
            batch_y = train_y[ n*batch_size : (n+1)*batch_size ]
            
            # Run the session
            sess.run( train_op, 
                      feed_dict={inputs: batch_x, 
                                 outputs: batch_y} )
            
            # Calculate cost
            c = sess.run(cost_op, 
                         feed_dict={inputs: batch_x, 
                                    outputs: batch_y} )            
            cost += c
        cost /= num_batches
        
        # Append the normalized cost to the cost list
        costs.append( cost )
        
        # Store the epoch number
        epochs.append( epoch )
            
        # Inform us after every 1000 epochs
        if epoch % 1000==0:
            print( "Epoch: %d - Error diff: %1.8f" %(epoch, cost) )
            
            # Convergence criteria
            if epoch > max_epochs  or abs(last_cost - cost) < tolerance:
                print( "--- STOPPING ---" )
                break
            
            # Store previous cost
            last_cost = cost
        
        # Increase epoch by 1
        epoch += 1
    
    '''Epoch: 0 - Error diff: 562.20129395
        Epoch: 1000 - Error diff: 18.38493919
        Epoch: 2000 - Error diff: 6.23961639
        Epoch: 3000 - Error diff: 3.43527198
        Epoch: 4000 - Error diff: 2.36873627
        Epoch: 5000 - Error diff: 2.00288463
        Epoch: 6000 - Error diff: 1.14719629
        Epoch: 7000 - Error diff: 0.97747177
        Epoch: 8000 - Error diff: 0.69312584
        Epoch: 9000 - Error diff: 0.65924841
        Epoch: 10000 - Error diff: 0.61906409
        Epoch: 11000 - Error diff: 0.44832635
        Epoch: 12000 - Error diff: 0.41447562
        Epoch: 13000 - Error diff: 0.36401308
        Epoch: 14000 - Error diff: 0.29663143
        Epoch: 15000 - Error diff: 0.22950944
        Epoch: 16000 - Error diff: 0.21943440
        Epoch: 17000 - Error diff: 0.21698588
        Epoch: 18000 - Error diff: 0.25753972
        Epoch: 19000 - Error diff: 0.24793650
        Epoch: 20000 - Error diff: 0.21406057
        Epoch: 21000 - Error diff: 0.16137363
--- STOPPING ---'''
    # Training has completed...
    print( "Test Cost =", sess.run(cost_op, 
                                   feed_dict={inputs: test_x, 
                                              outputs: test_y}) )
    '''Test Cost = 11.516622'''
    
    # Compute the predicted output for test_x
    pred_y = sess.run( yout, feed_dict={inputs: test_x, outputs: test_y} )
    
    for (y, yHat ) in zip( test_y, pred_y ):
        print( "%1.1f\t%1.1f" % (y, yHat ) )
        '''
        22.8    23.6
        16.1    19.2
        20.0    22.4
        17.8    20.4
        14.0    17.8
        19.6    22.0
        16.8    20.9
        21.5    23.3
        18.9    26.0
        7.0    8.3
        21.2    24.3
        '''
# Calculate R2 Score to evaluate the performance of the classifier
r2 =  metrics.r2_score(test_y, pred_y) 
print( "mean squared error = ", 
       metrics.mean_squared_error(test_y, pred_y))
print( "r2 score (coef determination) = ", 
       metrics.r2_score(test_y, pred_y))

'''
mean squared error =  11.643617949436841
r2 score (coef determination) =  0.8412243655290297
'''
# Let's plot regression 
#Directory to store the results figures
path2save = 'E:/PyDevWorkSpaceTest/AIwithTF/Chapter_4_AI_Fintech/BostonResults/'
fig = plt.figure()
xmin = min(test_y) 
xmax = max(test_y) + 5
plt.xlim(xmin, xmax)

x = np.linspace( xmin, xmax )
plt.scatter( test_y, pred_y )
plt.plot( x, x )

plt.text(5, 50, r'r2 = %1.4f' % r2)
plt.xlabel( "Test y" )
plt.ylabel( "predicted y" )
plt.title( "Prediction vs. Actual Y" )
plt.show()
fig.savefig(path2save+'PredVsRealBoston.png', bbox_inches='tight')

fig = plt.figure()
plt.scatter( test_y, - test_y + pred_y )
plt.axhline(0, color='black')
plt.xlabel( "Test y" )
plt.ylabel( "Test y - Predicted Y" )
plt.title( "Residuals" )
plt.show()
fig.savefig(path2save+'ResidualsBoston.png', bbox_inches='tight')

fig = plt.figure()
plt.semilogy( epochs, costs )
plt.xlabel( "Epochs" )
plt.ylabel( "Cost" )
plt.title( "Cost vs. Epochs")
plt.show()
fig.savefig(path2save+'CostVsEpochs.png', bbox_inches='tight')