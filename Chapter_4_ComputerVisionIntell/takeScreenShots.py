'''
Created on 25-Sep-2018

@author: DX
'''

# We will use pyautogui to grab the screenshot
import pyautogui 

# OpenCv will be used to preprocess the image and write it on the disk
import cv2

# Pynput will help us to take key inputs
from pynput import keyboard

# Numpy for array related operations
import numpy as np

# Let's create some global variables which we will use during 
# The image capturing

# I -- as a counter
# RESPONSE -- will be the key binding
# IMAGES -- will store the screenshots in a list  
global i,response,images 

# Initialize the global variables
i = 0 
images = []
response = []

# Here we define the bounding box coordinates to crop the screenshot  
cords = [520,570]
h,w = 490,740

# Function for cropping the images
def crop(im):    
    cropped = im[cords[0]:(cords[0]+h),cords[1]:(cords[1]+w)]
    
    # We will store the cropped image with the defined size
    cropped = cv2.resize(cropped,(2*96,2*64))
    return cropped

# A look up will convert the key response into numerical classes
def key_lookup(key):
    
    if key==keyboard.Key.up:key_code = 0    
    elif key==keyboard.Key.left:key_code = 1
    elif key==keyboard.Key.right:key_code = 2    
    elif key==keyboard.Key.down:key_code =  3
    elif key==keyboard.Key.esc:key_code =  4
    else: key_code = -1
        
    return key_code

# Now whenever a key is pressed by the user, the following 
# program will execute
def on_press(key): 
    
    # Initialize the global variables
    global i   
    global response
    global images         
    
    key_code = key_lookup(key)
    
    if key_code!= -1:
        print(key_code)
        # Increase the counter
        i = i+1   
        
        # Take a screenshot of the game window 
        pic = pyautogui.screenshot()
        
        # convert it into a numpy array
        im = np.array(pic)
        
        # As images return by pyautogui are in BGR format
        # we need to convert them into RGB format to store.
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        
        # Crop the meaningful area from the image
        im = crop(im)
        
        # Append the image and the associated key for that image into
        # different lists
        images.append(im)
        response.append(key_code)
        
        # Store the cropped image on the disk
        cv2.imwrite('TrainingScreens/'+str(i)+'.png',im)    
          

# When the user ends the process, the following function will be executed         
def on_release(key):    
    
    # When the user presses the 'esc' key, data collection will end
    # and a list of Images and key responses will be stored to the disk
    if key == keyboard.Key.esc:
        print('screen shot Stopped...')
        np.save('TrainingScreens/response.npy',response)
        np.save('TrainingScreens/Images.npy',images)
        return False

# Here we will look for key press    
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:    
    listener.join() 