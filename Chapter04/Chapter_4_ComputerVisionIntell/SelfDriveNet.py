'''
Created on 29-Sep-2018

@author: DX
'''

# Import different Keras layers to create the network
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout

# We will build a Sequential model without feedback
from keras.models import Sequential

# Here goes our network function
def Network(height,width,depth,weights=None):
    
    # Height: Image Height
    # Width: Image Width
    # Depth: Image Depth
    # Weights: learned weights
    
    # we will create this network with 3 classes
    classes = 3
    
    # Let's create the network 
    model = Sequential()
    
    # Add 2 Convolution layers with 'relu' activations
    # We will add one max pooling layer
    model.add(Conv2D(64,kernel_size=(3,3), 
                     activation='relu',
                     padding='same',
                     input_shape=(depth,height,width)))
    model.add(Conv2D(64,kernel_size=(3,3), 
                     activation='relu',padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    # Add 4 more Convolution layers with one max pool
    model.add(Conv2D(128,kernel_size=(3,3), 
                     activation='relu',padding='same'))
    model.add(Conv2D(128,kernel_size=(3,3), 
                     activation='relu',padding='same'))
    model.add(Conv2D(128,kernel_size=(3,3), 
                     activation='relu',padding='same'))
    model.add(Conv2D(128,kernel_size=(3,3), 
                     activation='relu',padding='same'))
    model.add(MaxPool2D(2,2))
    
    # Add 4 More Convolution layers with another max pool
    model.add(Conv2D(256,kernel_size=(3,3), 
                     activation='relu',padding='same'))
    model.add(Conv2D(256,kernel_size=(3,3), 
                     activation='relu',padding='same'))
    model.add(Conv2D(256,kernel_size=(3,3), 
                     activation='relu',padding='same'))
    model.add(Conv2D(256,kernel_size=(3,3), 
                     activation='relu',padding='same'))
    model.add(MaxPool2D(2,2))
    
    # Here we will convert 2D feature maps to 1D 
    model.add(Flatten())
    
    # Add a fully connected Layer with 50% drop-out
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.50))
    
    # One more fully connected layer goes here
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.50))
    
    # The final layer has 3 outputs
    model.add(Dense(classes,activation='softmax'))

    # If learned weights are provided, load them into the network        
    if weights is not None:
        model.load_weights(weights)
    
    return model