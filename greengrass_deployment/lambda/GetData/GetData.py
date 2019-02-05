
import greengrasssdk
import cv2
import numpy as np
import os
import boto3
import time
from datetime import datetime

s3 = boto3.client('s3')

os.environ["DISPLAY"] = ":0"

#PATH =
#S3_BUCKET =

# Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')

# CV2 Frames
CAPTURE_INDEX = 1
WINDOW_NAME = 'image'
CV_WAIT_KEY = 25
NUM_FRAMES = 5000

def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (100,100))
    return frame

def function_handler(event, context):
    
    cap = cv2.VideoCapture(CAPTURE_INDEX)
    time.sleep(1)  
    window = cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    counter = 0
    
    while counter < NUM_FRAMES-1:
        
        array = np.zeros((1000,1,100,100))
        
        for i in range(0,1000):
            counter = counter + 1
            ret, frame_read = cap.read()
            frame = preprocess(frame_read)
            array[i,0,:,:] = frame
            time.sleep(0.5)
        
        now = datetime.now()   
        filename = now.strftime("%d%m%y-%H-%M-%S")
        filepath = PATH + filename     
        np.save(filepath, array)
        s3.upload_file(filepath + ".npy", S3_BUCKET, filename+".npy") 
    cap.release()

    return

