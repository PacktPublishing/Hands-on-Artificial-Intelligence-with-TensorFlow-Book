'''
Created on 29-Sep-2018

@author: DX
'''

# Import numpy for matrics algebra
import numpy as np

# Import optimizer for cost function optimization 
from keras.optimizers import adam

# Import our network file
from SelfDrivingCar import SelfDriveNet

# Import model checkpoint to store model with conditions
from keras.callbacks import ModelCheckpoint

# Import data and label array from the disk  
path2data = 'Data/data.npy'
path2labels = 'Data/labels.npy'

# Path to store the model on the disk
path2save = 'Data/SelfDriveCar.h5'

# Path to store the model performance
model_performance = 'Data/History.npy'

# Let's load the data and label from the disk
data = np.load(path2data)
labels = np.load(path2labels)

# Define size
height = 64
width = 96
depth = 3

# Call the network file
model = SelfDriveNet.Network(height,width,depth)

# Create a model check point
# We will monitor the validation accuracy and store 
# the model for the highest accuracy 
call_back = ModelCheckpoint(path2save, monitor='val_acc', 
                            save_best_only=True,
                            save_weights_only=True,mode='auto')
# Here we will use ADAM optimizer with a starting learning rate
opt = adam(lr=1e-5)

# Compile the model with defined optimizer
# we will use log based cost function
model.compile(optimizer=opt,loss='categorical_crossentropy',
              sample_weight_mode='auto',metrics=['accuracy'])

# Now let's start training the network
history = model.fit(data, labels, 
          batch_size=32, epochs=100, 
          verbose=2, callbacks=[call_back], 
          validation_split=0.2)

# Store the model performance on the disk
np.save(model_performance,history.history)
print('Model Stored Successfully')

# train_data,test_data,train_label,test_label = \
# train_test_split(data,labels,test_size=0.2)


# train_gen = ImageDataGenerator(shear_range=0.2,                                                    
#                          zoom_range=0.2,
#                          featurewise_center = True,
#                          horizontal_flip=True,                         
#                          width_shift_range=0.2,
#                          height_shift_range=0.2,
#                          fill_mode='nearest')
# test_gen = ImageDataGenerator(featurewise_center = True)

# train_gen.mean = np.array(image_mean, dtype=np.float32).reshape((3, 1, 1))
# test_gen.mean = train_gen.mean

# test_gen.fit(test_data)
# train_gen.fit(train_data)
# 
# history = model.fit_generator(train_gen.flow(train_data,train_label,
#                                    batch_size=32,                                
#                                 #save_to_dir=aug_dir
#                                    ), 
#                     epochs=250, 
#                     verbose=2, callbacks=[call_back], 
#                     validation_data=test_gen.flow(test_data,test_label)
#                     )

