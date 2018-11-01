'''
Created on 24-May-2018

@author: Ankit Dixit
'''

#Pickle will be use to load and store the data
from _pickle import load
from pickle import dump

#OS will help us to get files from the directory
from os import listdir

#String will handle text processing operations
import string

#For using VGG architecture, Keras will be used 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

#Callbacks will be use to choose model configurations
from keras.callbacks import ModelCheckpoint

#Follwing will help us to create and loading Models 
from keras.models import Model, load_model

#We will use Keras for various image processing tasks
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

#Keras will help us in text related operations too
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

#We will use Keras utilities for one hot encoding
from keras.utils.np_utils import to_categorical

#For visualization of model created
from keras.utils.vis_utils import plot_model

#Microsoft's NLTK will help us to measure the performance of models
from nltk.translate.bleu_score import corpus_bleu

#Finally numpy for algebra operations
from numpy import argmax

#Here we will define our data directory
data_dir = 'E:/FlickerData/'

# Extract features from each image in the directory
def extract_features(directory):
    
    #DIRECTORY: is the path of image data
    
    # Load the VGG-16 model from keras
    model = VGG16()
    
    # Remove final layer so we can have features at the end
    model.layers.pop()
    
    # Re-structure the model after modification
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    
    # Let's see how our modified model looks
    print(model.summary())
    
    # In following loop we will extract feature from images
    # and create a dictionary so key of the dictionary will 
    # image name and key value will be extracted feature vector  
    
    #This is our feature directory
    features = dict()
    
    #Let's run the loop
    for name in listdir(directory):
        
        # Load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        
        # Convert the image pixels to a numpy array
        image = img_to_array(image)
        
        # Reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1],
                                image.shape[2]))
        
        # Now we will prepare the image for the VGG model
        # this pre-processing will convert images in compatible
        # form for VGG network.
        image = preprocess_input(image)
        
        # Now its time to extract the features
        feature = model.predict(image, verbose=0)
        
        # Once we got the features, we will put it into the dictionary
        image_id = name.split('.')[0]
        features[image_id] = feature
        #print('>%s' % name)
    
    # Return final feature vector    
    return features


# Here we will define our data directory
data_dir = 'E:/FlickerData/'

# Let's run above function to extract features from all images
directory = data_dir+'Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))

# Dump the extracted features on the disk
dump(features, open(data_dir+'features.pkl', 'wb'))


# Following function will load the text file
def load_doc(filename):
    
    #FILENAME: is the name of image sicription file
    
    # We will open the file in read only mode
    # so that we can not alter the content. 
    file = open(filename, 'r')
    
    # Read the contents
    text = file.read()
    
    # Close the original file
    file.close()
    
    # Return the descriptions
    return text

# Extract descriptions for images
def load_descriptions(doc):
    
    #DOC: loaded document 
    
    # Create empty dictionary to hold descriptions
    mapping = dict()
    
    # Start process each line 
    for line in doc.split('\n'):
        
        # Split line by white space it will give us 
        # each word separately in a list like structure 
        tokens = line.split()
        
        #We will not consider a line with a single character
        if len(line) < 2:
            continue
        
        # As each line have first word as image id, 
        # We will break the line in two part,
        # Image id and description 
        image_id, image_desc = tokens[0], tokens[1:]
        
        # Remove file extension from image name 
        image_id = image_id.split('.')[0]
        
        # Join all the word back and form the line
        image_desc = ' '.join(image_desc)
        
        # Create key if not present
        if image_id not in mapping:
            mapping[image_id] = list()
        
        # Insert descriptions as key value in respected keys
        mapping[image_id].append(image_desc)
    
    #Return dictionary for further processing
    return mapping

#Following function will clean the pre process the descriptions
def clean_descriptions(descriptions):
    
    #DESCRIPTIONS: is the dictionary contains image descriptions
    
    # We prepare a translation table which contains all the punctuation
    table = str.maketrans('', '', string.punctuation)
    
    # Let's iterate with different descriptions
    for key, desc_list in descriptions.items():
        
        # Here we will work on each description
        for i in range(len(desc_list)):
            
            # Extract description
            desc = desc_list[i]
            
            # Split it into words (tokenize)
            desc = desc.split()
            
            # Convert to all letters in lower case
            desc = [word.lower() for word in desc]
            
            # Remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            
            # Remove words with single character, like 
            # hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1]
            
            # Remove tokens with numeric values
            desc = [word for word in desc if word.isalpha()]
            
            # Join all the words again and store them
            desc_list[i] =  ' '.join(desc)

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    
    #DESCRIPTIONS: Cleaned descriptions dictionary
    #FILENAME: filename to store description onto disk
    
    #Create a list to store different lines
    lines = list()
    
    #Iterate to each descriptions
    for key, desc_list in descriptions.items():
        
        #Put the description with image name in different line
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    
    #Join all the lines in single document 
    data = '\n'.join(lines)
    
    #Open a file with defined filename with write permissions  
    file = open(filename, 'w')
    
    #Write the data into the file
    file.write(data)
    
    #Close the file
    file.close()

#File name to store cleaned descriptions
filename = data_dir+'Flickr8k_text/Flickr8k.token.txt'

#Load descriptions using 'load_doc'
doc = load_doc(filename)

#Parse descriptions using 'load_descriptions'
descriptions = load_descriptions(doc)

#Print the size of loaded descriptions
print('Loaded: %d ' % len(descriptions))

#Here we will clean descriptions
clean_descriptions(descriptions)

#Save descriptions to file
save_descriptions(descriptions, data_dir+'descriptions.txt')

# load a pre-defined list of image names
def load_set(filename):
    
    #FILENAME: is file name which contains image names
    
    #Load file from disk
    doc = load_doc(filename)
    
    #Create an empty list to store image names
    dataset = list()
    
    #Process line by line
    for line in doc.split('\n'):
        
        # Skip empty lines
        if len(line) < 1:
            continue
        
        # Get the image name
        identifier = line.split('.')[0]
        dataset.append(identifier)
    
    #Return subset 
    return set(dataset)

# Load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    
    #FILENAME: file name of cleaned descriptions
    #DATASET: is file with image names created by load_set
    
    # Load cleaned descriptions
    doc = load_doc(filename)
    
    #Create empty list to store sub set of description 
    descriptions = dict()
    
    #Let's iterate with different lines
    for line in doc.split('\n'):
        
        # Create words by splitting lines
        tokens = line.split()
        
        # Split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        
        # Skip images not in the set
        if image_id in dataset:
            
            # Create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            
            # Put start and end flag to each description
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            
            # Store modified description
            descriptions[image_id].append(desc)
    
    #Return descriptions 
    return descriptions

# load photo features
def load_photo_features(filename, dataset):
    
    #FILENAME: filename containing image features
    
    #load all features
    all_features = load(open(filename, 'rb'))
    
    #Create subset
    features = {k: all_features[k] for k in dataset}
    return features

#Load training dataset 
filename = data_dir+'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

#Create description subset of training description 
train_descriptions = load_clean_descriptions(data_dir+'descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

#Create feature sub set for training images
train_features = load_photo_features(data_dir+'features.pkl', train)
print('Photos: train=%d' % len(train_features))

# Convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    
    #DESCRIPTIONS: descriptions dictionary
    
    #Create an empty list to store descriptions
    all_desc = list()
    
    #Iterate through the disctionary keys and extract all descriptions
    for key in descriptions.keys():
        
        #Append all description one by one
        [all_desc.append(d) for d in descriptions[key]]
    
    #Return the description list
    return all_desc

# Fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    
    #DESCRIPTIONS: descriptions dictionary
    
    #Convert all descriptions in the list form
    lines = to_lines(descriptions)
    
    #Create tokenizer object from Keras
    tokenizer = Tokenizer()
    
    #Here we will fit a tokenizer on text data     
    tokenizer.fit_on_texts(lines)
    
    #Return tokenizer
    return tokenizer

#prepare tokenizer

#We will call create tokenizer here for training descriptions
tokenizer = create_tokenizer(train_descriptions)

#This is our vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

#Store the tokenizer for re use 
dump(tokenizer, open(data_dir+'tokenizer.pkl', 'wb'))

#Create multiple sequences from description data
def create_sequences(tokenizer, max_length, descriptions, photos):
    
    #TOKENIZER: to convert words into numerical values
    #MAX_LENGTH: length of largest sentence for our classifier
    #DESCRIPTIONS: image descriptions
    #PHOTOS: image features    
    
    #Let's start with creating empty list to store our sequences
    #for both input features and output word for each input
    X1, X2, y = list(), list(), list()
    
    # Walk through each image identifier
    for key, desc_list in descriptions.items():
        
        # Walk through each description for the image
        for desc in desc_list:
            
            # Here we will encode our sequences
            seq = tokenizer.texts_to_sequences([desc])[0]
            
            # Split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                
                # Split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                
                # Pad input sequence so every sequence will have equal length 
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                
                # Encode output sequence using one hot encoding
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                
                # Store input and output sequences into list
                X1.append(photos[key][0])#Image
                X2.append(in_seq)# Input Descriptions
                y.append(out_seq)# Output Descriptions
    
    #Return sequences            
    return np.array(X1), np.array(X2), np.array(y)

# Calculate the length of the description with the most words
def max_length(descriptions):
    #DESCRIPTIONS: image descriptions
    
    #Convert descriptions in lines
    lines = to_lines(descriptions)
    
    #And measure the length of a sequence
    return max(len(d.split()) for d in lines)

# Import different types of layers from keras
from keras.layers import*

# Define the captioning model
def define_model(vocab_size, max_length):
    
    #VOCAB_SIZE: Number of unique words
    #MAX_LENGTH: maximum length of sequence
    
    # Feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.25)(inputs1)
    
    #Compress the size of 4096 feature vector to 256
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence model
    inputs2 = Input(shape=(max_length,))
    
    # Add a word embedding layer to learn vector representation
    # of that word
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.25)(se1)
    
    # Add a LSTM layer to process the word vector
    # LSTM will predict next word of the sequence
    # Here it is independent of image. 
    se3 = LSTM(256)(se2)
    
    # Decoder model
    # This model will responsible to predict a word against
    # an image and a sequence.
    
    # First we will combine a image and a sequence
    # predicted by LSTM
    decoder1 = add([fe2, se3])
    
    # And we will process the combined information here
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Here we will define input and output variable layers of our model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    
    #We will compile our model with cross entropy loss and
    #Adaptive momentum variant of stochastic gradient descent
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # Let's our model parameters
    print(model.summary())
    
    #Plot model to visualize it and store on to local disk
    plot_model(model, to_file=data_dir+'model.png', show_shapes=True)
    
    #Return the created model
    return model


# Determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# Prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, 
                                            max_length, 
                                            train_descriptions, 
                                            train_features)
 
# Dev dataset
 
# Load test set
filename = data_dir+'Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))

# Descriptions
test_descriptions = load_clean_descriptions(data_dir+'descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))

# Image features
test_features = load_photo_features(data_dir+'features.pkl', test)
print('Photos: test=%d' % len(test_features))

# Prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, 
                                         max_length, 
                                         test_descriptions, 
                                         test_features)
 
# Let's fit our model 
# First we will create our model
model = define_model(vocab_size, max_length)

# Define checkpoint callback so we can store model onto disk
# we will store our model whenever validation loss of model
# will less than previous loss.
filepath = data_dir+'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                             verbose=1, save_best_only=True, mode='min')

# Fit model our model for 20 epochs 
model.fit([X1train, X2train], ytrain, epochs=20, 
          verbose=2, callbacks=[checkpoint], 
          validation_data=([X1test, X2test], ytest))

# Map an integer to a word
def word_for_id(integer, tokenizer):
    
    #INTEGER: Maximum probability index
    #TOKENIZER: tokenizer for word mapping
    
    #Search the word using index value  
    for word, index in tokenizer.word_index.items():
        
        #If found a match return the word
        if index == integer:
            return word
    #If no match found retrun NONE    
    return None
 
# Generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    
    #MODEL: trained caption generator model
    #TOKENIZER: to map word to integer and integer to word
    #PHOTO: Image features
    #MAX_LENGTH: maximum sequence length with model trained earlier
    
    # Our initial word for the sequence 
    in_text = 'startseq'
    
    # Start iterating for the sequence
    for i in range(max_length):
        
        # Integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        
        # Pad input 
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict next word with image and previous sequence as input
        yhat = model.predict([photo,sequence], verbose=0)
        
        # Get maximum probability index
        yhat = argmax(yhat)
        
        # Convert index into word using tokenizer
        word = word_for_id(yhat, tokenizer)
        
        # Stop is there is no predictions
        if word is None:
            break
        
        # Append predicted words to the sequence
        in_text += ' ' + word
        
        # Stop if we predict the end of the sequence
        if word == 'endseq':
            break
    #Return the generated caption
    return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
  
# prepare test set
 
# load test set
filename = data_dir+'Flickr8k_text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))

# descriptions
test_descriptions = load_clean_descriptions(data_dir+'descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))

# photo features
test_features = load_photo_features(data_dir+'features.pkl', test)
print('Photos: test=%d' % len(test_features))
 
# load the model
filename = data_dir+'model-ep002-loss3.245-val_loss3.612.h5'
model = load_model(filename)

# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

# Load the tokenizer
tokenizer = load(open(data_dir+'tokenizer.pkl', 'rb'))

# Define the max sequence length (from training)
max_length = 34

# Load the model
model = load_model(data_dir+'model-ep004-loss3.468-val_loss3.818.h5')

# Load and prepare the visual features
photo = extract_features(data_dir+'example.jpg')

# Generate description
description = generate_desc(model, tokenizer, photo, max_length)

#Import opencv library to load original image
import cv2

#Read Image
im = cv2.imread('X:/Test/example.jpg')

#Put Description on the Image
im = cv2.putText(im,description,(5,300),cv2.FONT_HERSHEY_DUPLEX,0.6,(0,0,0))

# Write image into the directory
cv2.imwrite('X:/Test/example_result.jpg',im)

# Show Image
cv2.imshow('',im)
cv2.waitKey()

#Print description
print(description)
