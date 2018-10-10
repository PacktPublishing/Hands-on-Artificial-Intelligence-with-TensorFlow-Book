'''
Created on 29-Sep-2018

@author: DX
'''

# We will use pandas to visualize class distribution
# and apply some pre-processing on the data set
import pandas as pd

# Matplotlib will help us to plot the class distribution
import matplotlib.pyplot as plt

# Numpy for calculating the mean of the data set
import numpy as np

# Let's first define path of labels and 
# Images along with the path to save processed data.
path2data = 'Data/Images.npy'
path2label = 'Data/response.npy'
path2save = 'Data/'
           
# Following function will create the categorical data set           
def create_labels(labels,num_classes):
    
    cat_labels = np.zeros((len(labels),
                           num_classes),dtype="uint8")
    
    for i in range(len(labels)):        
        cat_labels[i,labels[i]] = 1
    
    return cat_labels

# Let's load the label data first
labels = np.load(path2label)

# Remove the last index (belongs to 'esc' key)
labels = labels[0:-1]

# We will create a Pandas DataFrame for labels
df = pd.DataFrame(labels,columns=['Action'])

# Let's plot the histogram for the class distribution
# for 4 classes
df.hist(bins=4)
plt.show()

# We need to remove at least 60% samples from class 0
# to balance the data set following lines will give us
# indices to be removed, it will be selected randomly  
idx_up = df.query('Action == 0').sample(frac=.6).index

# We need to remove all the instances for class 3
# as there occurrence is very rare in the data set
idx_down = df.query('Action == 3').sample(frac=1.0).index

# We will convert indices into list
# and merge indices to be removed from both the classes 
list_index_up = idx_up._data.tolist()
list_index_down = idx_down._data.tolist()
list_index = list_index_up + list_index_down

# Convert the labels to list from the DataFrame
labels = df['Action'].tolist()

# Load the images which required to be removed 
images = np.load(path2data)
images = images[0:-1] # Remove last ('esc') index
 
# We will create two new list for updated indices
new_labels = []
new_images = []

# In following loop we will remove the images and labels 
for i in range(len(labels)):
    if i not in (list_index):
        new_labels.append(labels[i])
        new_images.append(images[i])

# Convert list into numpy array  
images = np.array(new_images)
labels = np.array(new_labels)

# Let's visualize updated labels 
df = pd.DataFrame(labels)
df.hist(bins=3)
plt.show()

# Calculate data mean 
im_mean_all = np.mean(images,0)
im_mean = [np.mean(np.mean(im_mean_all[:,:,2])),
           np.mean(np.mean(im_mean_all[:,:,1])),
           np.mean(np.mean(im_mean_all[:,:,0]))]

# Subtract the mean from the images to normalize the data
image_data = images - im_mean
# Reshape image data so in NX3X64X96 form
image_data = np.transpose(image_data,[0,3,1,2])

print('Size of the image data set:', image_data.shape)

# Here we will convert label into one hot encoded array
cat_labels = create_labels(labels,3)

# Store pre processed data on the disk along with the mean
np.save(path2save+'data.npy',np.array(image_data))
np.save(path2save+'labels.npy',np.array(cat_labels))
np.save(path2save+'mean.npy',np.array(im_mean))

print('Done')