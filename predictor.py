


import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
import numpy as np
import cv2
from sklearn.utils import shuffle
from PIL import Image
import imutils
import os

# global variables
bg = None

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    silhouette = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(silhouette.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (silhouette, segmented)

def main():
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 100, 300, 300, 500

    # initialize num of frames
    num_frames = 0
    start_recording = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width = 700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                (silhouette, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (255, 255, 255))
                if start_recording:
                    cv2.imwrite('Temp.jpg', silhouette)

                    predictedClass, confidence = getPrediction()
                    #showStatistics(predictedClass, confidence)
                    cv2.putText(clone,"Detected Gesture : " + str(predictedClass), (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(clone,"Probability : " + str(confidence * 100) + "%", (100, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


                cv2.imshow("S", silhouette)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (255,255,255), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            os.remove("Temp.jpg")
            break
        
        if keypress == ord("s"):
            start_recording = True

def getPrediction():

    img = Image.open('Temp.jpg')
    img = img.resize((50, 50))
    img = np.array(img, dtype=np.float32)
    img /= 255.
    print(img.shape)
    img = np.expand_dims(img,axis=2)
    img = np.expand_dims(img,axis=0)
    prediction = model.predict(img)
    return np.argmax(prediction), np.amax(prediction) 

# Model defined
model = Sequential([
              tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
              tf.keras.layers.MaxPool2D(pool_size=(2,2)),
              tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
              tf.keras.layers.MaxPool2D(pool_size=(2,2)),
              tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
              tf.keras.layers.MaxPool2D(pool_size=(2,2)),
              tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
              tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
              tf.keras.layers.MaxPool2D(pool_size=(2,2)),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dropout(0.3),
              tf.keras.layers.Dense(units=256, activation='relu'),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(units=64, activation='relu'),
              tf.keras.layers.Dense(units=10, activation='softmax')])

model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])

# Load Saved Model
model.load_weights("my_checkpoint")

main()
