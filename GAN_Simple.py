'''
Created on 07-May-2018

@author: aii32199
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


seed = 42
np.random.seed(42)
tf.set_random_seed(seed)

def RealInput(N,mu=0,sigma=1):
    p = np.random.normal(mu,sigma,N)
    return np.sort(p) 

def RandomInput(nrange,N):
    return np.linspace(-nrange, nrange, N)+np.random.random(N)*0.01 

#Following function is implementation of a Perceptron layer 
def Layer(inp,out_dim,scope,stddev=1.0):
    
    #This function will have input sample INP;
    #User need to mention the dimension of the output;
    #STDDEV will be used to initialize the weight variable.
    with tf.variable_scope(scope or 'Layer'):
        
        #First we will create a tensorflow variable for weights.
        #Its dimension will be calculated on the basis of INP and OUT_DIM
        #We will initialize the weights using a gaussian distribution  
        w = tf.get_variable('w',
                            [inp.get_shape()[1],out_dim], 
                            initializer=
                            tf.random_normal_initializer(stddev=stddev))
        
        #Then we will create a tf variable for bias.
        b = tf.get_variable('b',[out_dim],
                            initializer=tf.constant_initializer(0.0))
        
        #Finally we will multiply weight matrix to the INP and add the bias term
        return tf.matmul(inp,w)+b

#Following function is implementation of discriminator
def Discriminator(inp,hlayer_dim):
    
    #INP is input to the function
    #HLAYER_DIM is number of weights required in the layer 
    
    #This is our first linear layer with the ReLu transformation  
    L_0 = tf.nn.relu(Layer(inp, hlayer_dim, 'D_L_0'))
    
    #Second layer
    L_1 = tf.nn.relu(Layer(L_0, hlayer_dim,'D_L_1'))
    
    #Third layer
    L_2 = tf.nn.relu(Layer(L_1, hlayer_dim,'D_L_2'))
    
    #Final layer of the network will have a Sigmoid transformation 
    #we have used Sigmoid to limit our output in the range of 0 and 1     
    out = tf.sigmoid(Layer(L_2, 1,'D_L_3'))
    
    #Finally function will return the output of final layer 
    return out

#This is our fake sample generator
def Generator(inp,out_dim):
    
    #Input to this network are samples generated by RandomInput function.
    #This function will create a two layer generator model.
    
    #We will add a Linear layer, output of which will
    #be send to a soft-plus function to create non-linearity   
    L_0 = tf.nn.softplus(Layer(inp, out_dim, 'G_L_0'))    
    
    #At the output end we will just add a linear layer without
    #any transformation function.    
    out = Layer(L_0, 1,'G_L_1')
    
    return out


#Following is the function for optimizers
def optimizer(loss,var_list):
    
    #We will use initial learning rate as lr
    lr = 0.001
    
    #Update the weights after each iteration
    step = tf.Variable(0, trainable=False)  
    
    #We will use adaptive momentem to tune the weights
    #It will change the learning rate according to speed 
    #of convergence of the loss function.   
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,
                                                         global_step=step,
                                                         var_list=var_list)    
    return optimizer
    
#This function will return logaritimic of tensor
def log(x):
    
    #We will use tensorflow's log function for our task
    return tf.log(tf.maximum(x,1e-5))

#Following is the class written to create GAN
class GAN():
    
    #We will write a class for GAN for simplicity
    #in variable management
    #You can also write a simple function for the task.
    #Currently we will put a default batch size of 8 samples.
    def __init__(self,inp_size=8):
        
        #We will create two place holders to store Real sample
        #and random samples with the input size defined by the user.  
        self.z = tf.placeholder(tf.float32, shape=(inp_size, 1))
        self.x = tf.placeholder(tf.float32, shape=(inp_size, 1))
        
        #Whenever we are working with generator, 
        #we want all the parameters,
        #have same prefix 'G' so that we can identify them.
        with tf.variable_scope('G'):                
            
            #Here we will create our Generator with number of 
            #hidden weights defined by user. 
            self.G = Generator(self.z, 4)
        
        #Whenever we are working with discriminator, 
        #we want all the parameters, have same prefix 'D' 
        #so that we can identify them.
        with tf.variable_scope('D'):
            
            #Here we will create our Discriminator 1 with number of 
            #hidden weights defined by user, 
            #it will handle the real data samples.
            self.D_1 = Discriminator(self.x, 4)
        
        #Here we will set flag reuse=True, it will help us to use same 
        #parameters from Discriminator 1    
        with tf.variable_scope('D',reuse=True):
            
            #Here we will create our Discriminator 2
            #with number of hidden weights 
            #defined by user, it will handle the 
            #output of generator(Fake data).
            self.D_2 = Discriminator(self.G, 4)
                
        #Once we got outputs from both the discriminator we will use loss,
        #function for checking the performance of our classifiers
        
        #First we will check combined performance to update Discriminator.
        self.loss_d = tf.reduce_mean(-log(self.D_1)-log(1-self.D_2))
        
        #Then we will measure the loss for Generaor optimization.
        self.loss_g = tf.reduce_mean(-log(self.D_2))
        
        #Here we will create a list of all 
        #trainable parameters from discriminator
        #as well as generator
        var = tf.trainable_variables()
        
        #Here is the reasom we have defined different variable scopes,
        #for generator and discriminator, we will seperate the  
        #parameters of both the networks to optimize them individually 
        self.d_params = [v for v in var if v.name.startswith('D/')]
        self.g_params = [v for v in var if v.name.startswith('G/')]
        
        #Here our optimization will take place
        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)

def train(model,steps):
    
    #Let's choose a sample size of 8 samples per batch 
    N = 8
    
    #We will start with creating a tensorflow session        
    with tf.Session() as sess:        
        
        #Initialize all local and global variables
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        
        #Start training for user defined number of iterations
        for step in range(steps):
            
            #Get real samples  
            x = RealInput(N)
            
            #Get random samples
            z = RandomInput(N,N)
            
            #Start training for both the discriminators 
            loss_d,_, = sess.run([model.loss_d,model.opt_d],
                                {model.x: np.reshape(x, (N,1)),
                                 model.z: np.reshape(z, (N,1))})
            
            #Start training for Generator for random sample inputs
            z = RandomInput(N,N)
            loss_g,_, = sess.run([model.loss_g,model.opt_g],
                                {model.z: np.reshape(z, (N,1))})
            
            #Here we will print the loss values for each iteration
            print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))
        visualize(10000, sess, model)
 
def visualize(npoints,session,model):
    
    #NPOINTS are the number of sample points
    #SESSION is tensorflow session
    #MODEL of Generator 
    
    #Generate real samples for reference  
    x = RealInput(npoints)
    
    #Generate Random samples to input the generator in range of -8 to 8
    z = RandomInput(8, npoints)  
    
    #We will process 8 points in one pass      
    batch_size = 8
    
    #We will create a histogram with hundred bins in range of -8 to 8
    nbins = 100
    bins = np.linspace(-8, 8, nbins)
    
    #First create the histogram for real data points
    pd, _ = np.histogram(x, bins=bins, density=True)
    
    #In following loop we will calculate generator's response
    #for each batch of 8 points.
    
    #Initialize a variable for store the response
    g = np.zeros((npoints, 1))
    
    #Following loop will run for number of batches
    for i in range(npoints // batch_size):
        
        #Here we will calculate the response of G for each batch
        g[batch_size * i:batch_size * (i + 1)] = session.run(
            model.G,
            {
                model.z: np.reshape(
                    z[batch_size * i:batch_size * (i + 1)],
                    (batch_size, 1)
                )
            }
        )
    
    #Once we got the response of G for all sample points
    #create the histogram for generators output
    pg, _ = np.histogram(g, bins=bins, density=True)
    
    #Following lines will plot the Real distribution and fake
    #distribution on the same plot.
    p_x = np.linspace(-8, 8, len(pd))
    f, ax = plt.subplots(1)   
    ax.set_ylim(0, 1) 
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()
    
    
#Create a main() function to call our defined method       
def main():        
    
    #Create object of GAN class
    model = GAN()
    #Train out networks for 5000 iteration
    train(model, 5000)
    
       
#main()    