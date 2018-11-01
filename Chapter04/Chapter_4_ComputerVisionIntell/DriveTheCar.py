'''
Created on 29-Sep-2018

@author: DX
'''

# Import pyautogui to grab the screenshot
import pyautogui

# Import time to control the key press
import time 

# Open Cv will help us in image processing
import cv2

# Pynput will be used to control the keyboard 
from pynput import keyboard 
con = keyboard.Controller()

# numpy for array operations
import numpy as np

# Import our network
from SelfDrivingCar import CropImages, SelfDriveNet

# Turn Left
def left():
    print('left')        
    con.press(keyboard.Key.left)        
    time.sleep(0.1)    
    con.press(keyboard.Key.up)
    time.sleep(0.009)
    con.release(keyboard.Key.up)
    con.release(keyboard.Key.left)

# Turn Right
def right():
    print('right')
    con.press(keyboard.Key.right)        
    time.sleep(0.1)
    con.press(keyboard.Key.up)
    time.sleep(0.009)
    con.release(keyboard.Key.up)
    con.release(keyboard.Key.right)

# Move forward
def up():
    con.press(keyboard.Key.up)
    print('up')        
    time.sleep(0.07)
    con.release(keyboard.Key.up)

# Here is the drive function            
def drive():
    
    # Load the model here
    model_path = 'Data/SelfDriveCar.h5'
    drive_net = SelfDriveNet.Network(64,96,3,model_path)
    
    # Load data mean     
    im_mean = np.load('Data/mean.npy')
    
    # Now we will run a continuous loop to drive the car
    while(1):
        
        # Grab the screenshot
        pic = pyautogui.screenshot()
        
        # Convert it into an array and convert the color space    
        im = np.array(pic)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        
        # Crop the image to get the game screen
        crop = CropImages.crop(im)        
        
        # Subtract the mean from the image
        crop = crop.astype("float32")-im_mean
        temp = []
        temp.append(np.rollaxis(crop, axis=2))
        
        # Send the image to the network and get probabilities
        probs = drive_net.predict(np.array(temp))
        
        # Get the class using max probability index 
        prediction = np.argmax(probs,axis=1)
        
        # Take action 
        if prediction==0:up()
        if prediction==1:left()
        if prediction==2:right()

drive()        