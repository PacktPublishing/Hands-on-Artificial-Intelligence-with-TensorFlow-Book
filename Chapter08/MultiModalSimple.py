'''
Created on 24-May-2018

@author: Ankit Dixit
'''

#Import all the packages we will going to use 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Initialize the random number generator for reproducible results
np.random.seed(41)
tf.set_random_seed(41)

#Number of Sample points
n = 400

#Probability of distribution
p_c = 0.5

#Let's generate a binomial distribution for class variable
#It will divide the 50% samples in class 1 and 0 both
c = np.random.binomial(n=1, p=p_c, size=n)

#Our two information sources will be 2 bivariate Gaussian distribution
#So we need to define 2 mean value for each distribution

#Mean values for Visual data set
mu_v_0 = 1.0
mu_v_1 = 8.0

#Mean values for textual data set
mu_t_0 = 13.0
mu_t_1 = 19.0

#Now we will generate the two distributions here
x_v = np.random.randn(n) + np.where(c == 0, mu_v_0, mu_v_1)
x_t = np.random.randn(n) + np.where(c == 0, mu_t_0, mu_t_1)

#Let's normalize data with the mean value
x_v = x_v - x_v.mean()
x_t = x_t - x_t.mean()

#Visualize the two classes with the combine information
plt.scatter(x_v, x_t, c=np.where(c == 0, 'blue', 'red'))
plt.xlabel('visual modality')
plt.ylabel('textual modality');
plt.show()

#Define number of points in the sample set 
resolution = 1000

#Create a linear sample set from visual information distribution
vs = np.linspace(x_v.min(), x_v.max(), resolution)

#Create linear sample set from textual information distribution
ts = np.linspace(x_t.min(), x_t.max(), resolution)

#In following lines we will propagate these sample point to create
#proper data set, it will help to create pair of both the information 
vs, ts = np.meshgrid(vs, ts)

#Here we will flatten our arrays
vs = np.ravel(vs)
ts = np.ravel(ts)

#Let's start with creating variables

#It will store the visual information
visual = tf.placeholder(tf.float32, shape=[None])

#This will store textual information
textual = tf.placeholder(tf.float32, shape=[None])

#And the final one will be responsible for holding class variable
target = tf.placeholder(tf.int32, shape=[None])
 
#As we are working wit a binary problem 
NUM_CLASSES = 2

#We will use fixed number of neuron for every layer 
HIDDEN_LAYER_DIM = 1

#This is our Visual feature extractor,
#It will be responsible for extraction of useful features,
#from visual samples, we will use tanh as activation function.
h_v = tf.layers.dense(tf.reshape(visual, [-1, 1]),
                      HIDDEN_LAYER_DIM,
                      activation=tf.nn.tanh)

#This is our Textual feature extractor,
#It will be responsible for extraction of useful features,
#from visual samples, we will use tanh as activation function.
h_t = tf.layers.dense(tf.reshape(textual, [-1, 1]),
                      HIDDEN_LAYER_DIM,
                      activation=tf.nn.tanh)

#Now as we have features from both the sources,
#we will fuse the information from both the sources,
#by creating a stack, this will be our aggregator network 
fuse = tf.layers.dense(tf.stack([h_v, h_t], axis=1),
                    HIDDEN_LAYER_DIM,
                    activation=tf.nn.tanh)

#Flatten the data here 
fuse = tf.layers.flatten(fuse)

#Following layers are the part of the same aggregator network 
z = tf.layers.dense(fuse,HIDDEN_LAYER_DIM,activation=tf.nn.sigmoid)

#This one is our final dens layer which used to convert network output
#for the two class
logits = tf.layers.dense(z, NUM_CLASSES)

#We want probabilities at the output, sigmoid will help us
prob = tf.nn.sigmoid(logits)

#We will use Sigmoid cross entropy as the loss function
loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=
                                       tf.one_hot(target, depth=2),
                                       logits=logits)

#Here we optimize the loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss)

def train(train_op, loss,sess):
    
    #TRAIN_OP: optimizer
    #LOSS: calculated loss
    #SESS: Tensorflow session
    
    #First initialize all the variables created 
    sess.run(tf.global_variables_initializer())
    
    #We will monitor the loss through each epoch 
    losses = []
    
    #Let's run the optimization for 100 epochs
    for epoch in range(100):
        _, l = sess.run([train_op, loss], {visual: x_v,
                                           textual: x_t,
                                           target: c})
        losses.append(l)
    
    #Here we will plot the training loss
    plt.plot(losses, label='loss')
    plt.title('loss')
    

#Create a tensorflow session
sess = tf.Session()

#Start training of the network    
train(train_op, loss,sess)

#Run the session
zs, probs = sess.run([z, prob], {visual: vs, textual: ts})

def plot_evaluations(evaluation, cmap, title, labels):
    
    #EVALUATION: Probability op from network
    #CMAP: colormap options
    #TITLE: plot title 
    #LABELS: Class labels
    
    #First we will plot our distributions as we have done previously
    plt.scatter(((x_v - x_v.min()) * resolution / (x_v - x_v.min()).max()),
                ((x_t - x_t.min()) * resolution / (x_t - x_t.min()).max()),
                c=np.where(c == 0, 'blue', 'red'))
    
    #Give the titles to our plots with labeling the axes
    plt.title(title, fontsize=14)
    plt.xlabel('visual modality')
    plt.ylabel('textual modality')
    
    #Here we will create a color map to draw the boundaries
    plt.imshow(evaluation.reshape([resolution, resolution]),
               origin='lower',
               cmap=cmap,
               alpha=0.5)
    
    #Let's put a color bar to create a fancy looking plot
    cbar = plt.colorbar(ticks=[evaluation.min(), evaluation.max()])
    cbar.ax.set_yticklabels(labels)
    cbar.ax.tick_params(labelsize=13)

#We will plot the probabilities    
plot_evaluations(probs[:, 1],
                 cmap='bwr',
                 title='$C$ prediction',
                 labels=['$C=0$', '$C=1$'])

#Show the plots over here
plt.show()