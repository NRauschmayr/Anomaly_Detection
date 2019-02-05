import json
import os
import time
import logging
import numpy as np
import cv2
import greengrasssdk
import load_model
import mxnet as mx
import scipy
from scipy import signal
import threading

gg_client = greengrasssdk.client('iot-data')

# CV2 Frames
CAPTURE_INDEX = 1
INFINITE_FRAME = -1
CV_WAIT_KEY = 25

def display_frame(frame, diff, rec):
    frame = cv2.resize(frame, (200,200))
    diff = cv2.resize(diff, (200,200))
    rec = cv2.resize(rec, (200,200))
    cv2.imshow("anomalies", frame)
    cv2.imshow("difference", diff)
    cv2.imshow("reconstructed", rec)
    cv2.waitKey(CV_WAIT_KEY)

def preprocess(frame):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (100,100))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.expand_dims(gray, axis=0)
    gray = np.expand_dims(gray, axis=0)
    return gray/255.0, frame

def start_inference(model, cap):
    while True:
    	ret, frame_read = cap.read()
    	gray, frame = preprocess(frame_read)
        
    	image = mx.nd.array(gray)
    	image = image.as_in_context(mx.gpu())
    
    	rec_image = model(image)    
    	diff = mx.nd.abs(rec_image-image)
    	rec_image = rec_image.asnumpy()
    	diff = diff.asnumpy()
    	
    	H = scipy.signal.convolve2d(diff[0,0,:,:], np.ones((4,4)), mode='same') 
    
    	frame[H > 4] = [220,20,60]
    	threading.Thread(target=display_frame, args=[frame, diff[0,0,:,:], rec_image[0,0,:,:]]).start() 
    	

def function_handler(event, context):
    cap = cv2.VideoCapture(CAPTURE_INDEX)
    time.sleep(1)  

    model = load_model.load_model()
    start_inference(model, cap)

    return


