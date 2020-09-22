#!/usr/bin/env python

import sys
#import OpenCV
import cv2
import numpy as np
import servo
import ctypes 
from time import sleep

# contrast/brightness values
contrast_value    = 0
brightness_value  = 0

#mapping function for servo
def mapping(x,in_min,in_max,out_min,out_max):
    toReturn =  (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return toReturn

# Variables for keeping track of the current servo positions
servoTiltPosition = 90
servoPanPosition = 90

# pan/tilt servo ids
tiltChannel = 0
panChannel = 1

#PWM setup
pca9685 =PCA9685()
pca9685.openPCA9685()
pca9685.setAllPWM(0,0)
pca9685.reset() 
pca9685.setPWMFrequency(60)


# These variables hold the x and y location for the middle of the detected face
midFaceY=0
midFaceX=0

# The variables correspond to the middle of the screen, and will be compared to the midFace values
midScreenY = (height/2)
midScreenX = (width/2)
midScreenWindow = 10 # This is the acceptable 'error' for the center of the screen

# The degree of change that will be applied to the servo each time we update the position
stepSize=1

#Create a window for the sketch
size( 320, 240 )


#opencv = new OpenCV( this )
cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-  method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
cv2.capture( width, height ) # open video stream
cv2.cascade( OpenCV.CASCADE_FRONTALFACE_ALT ) # load detection description, here-> front face detection : "haarcascade_frontalface_alt.xml"

# print usage
print("Drag mouse on X-axis inside this sketch window to change contrast" )
print("Drag mouse on Y-axis inside this sketch window to change brightness" )


#Send the initial pan/tilt angles to look straight forward
pca9685.tiltChannel
pca9685.servoTiltPosition # Send the Tilt Position (currently 90 degrees)
pca9685.panChannel
pca9685.servoPanPosition # Send the Pan Position (currently 90 degrees)



# grab a new frame and convert to gray
cv2.convert( GRAY )
cv2.contrast( contrast_value )
cv2.brightness( brightness_value )
ret_val, frame = cap.read();
gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frameRs=cv2.resize(frame, (640,360))
grayRs=cv2.resize(gray_frame,(640,360))
face_cascade = cv2.CascadeClassifier('/home/nvidia/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
### Change grayRs to frameRs if you need face detector to work on RGB.
faces = face_cascade.detectMultiScale(grayRs, 1.3, 5)
for (x,y,w,h) in faces:
    grayRs = cv2.rectangle(grayRs,(x,y),(x+w,y+h),(255,0,0),2)
                    
                
    displayBuf = grayRs

# proceed detection
#Rectangle() faces = cv2.detect( 1.2, 2, cv2.HAAR_DO_CANNY_PRUNING, 40, 40 )

# display the image
image( cv2.image(), 0, 0 )

# draw face area(s)
noFill()
stroke(255,0,0)
for i in range(0,faces.length-1):	
        rect( faces(i).x, faces(i).y, faces(i).width, faces(i).height )

# Find out if any faces were detected
if faces.length > 0:
    #If a face was found, find the midpoint of the first face in the frame
    #The .x and .y of the face rectangle corresponds to the upper left corner of the rectangle
    #so we manipulate these values to find the midpoint of the rectangle
    midFaceY = faces(0).y + (faces(0).height/2)
    midFaceX = faces(0).x + (faces(0).width/2)

    #Find out if the Y component of the face is below the middle of the screen
    if midFaceY < (midScreenY - midScreenWindow):
        if pca9685.servoTiltPosition >= 5:
            pca9685.servoTiltPosition -= stepSize # If it is below the middle of the screen, update the tilt position variable to lower the tilt servo
    #Find out if the Y component of the face is above the middle of the screen        
    elif midFaceY > (midScreenY + midScreenWindow):
        if pca9685.servoTiltPosition <= 175:
            pca9685.servoTiltPosition +=stepSize # Update the tilt position variable to raise the tilt servo
    #Find out if the X component of the face is to the left of the middle of the screen        
    if midFaceX < (midScreenX - midScreenWindow):
        if pca9685.servoPanPosition >= 5:
            pca9685.servoPanPosition -= stepSize # Update the pan position variable to move the servo to the left
    #Find out if the X component of the face is to the right of the middle of the screen        
    elif midFaceX > midScreenX + midScreenWindow:
        if pca9685.servoPanPosition <= 175:
            pca9685.servoPanPosition +=stepSize  # Update the pan position variable to move the servo to the right

# Update the servo positions
pca9685.tiltChannel
pca9685.servoTiltPosition # Send the updated tilt position
pca9685.panChannel
pca9685.servoPanPosition # Send the updated pan position
sleep(1)









