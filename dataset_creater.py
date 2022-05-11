# 1430  CV group project
# this file have reference the idea from an online article written by Ilango
# Gogul Ilango. Hand gesture recognition using python and opencv - part 1, Apr 2017.
# https://gogul.dev/software/hand-gesture-recognition-p1



# organize imports
import cv2
import imutils
import numpy as np

# global variables
bg = None
cat_str = "temp"
image_num = 0
base = 0 
train_test = "train"

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=28):
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
        contour  = max(cnts, key=cv2.contourArea)
        return (silhouette, contour)


def main():
    global image_num
    global cat_str
    global base 
    global train_test

    aWeight = 0.5

    camera = cv2.VideoCapture(0)

    # region of hand detecting area
    top, right, bottom, left = 100, 100, 300, 300


    num_frames = 0
    last_reset = num_frames


    start_recording = False

    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()
        if (grabbed == True):

            # resize the frame
            frame = imutils.resize(frame, width=700)

            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            
            one_frame = frame.copy()

            # get the height and width of the frame
            (height, width) = frame.shape[:2]

            # get the ROI
            roi = frame[top:bottom, right:left]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            keypress = cv2.waitKey(1) & 0xFF

            # to reset press r
            if keypress == ord("r"):
                last_reset = num_frames
                print("reset start")

            if num_frames - last_reset < 30:
                run_avg(gray, aWeight)
                print(num_frames - last_reset)
                if (num_frames - last_reset== 29):
                    print("background set")
            else:
                # segment the hand region
                hand = segment(gray)

                if hand is not None:

                    (silhouette, contour) = hand

                    # draw contours 
                    cv2.drawContours(
                        one_frame, [contour + (right, top)], -1, (0, 255, 0))
                    if start_recording:

                        # Mention the directory in which you wanna store the images followed by the image name
                        print("reached" + str(image_num + base))
                        cv2.imwrite("/Users/wangsentian/Downloads/Hand/Dataset/" + train_test + "/"+ cat_str + "/" + cat_str + "_" + 
                                    str(image_num+base) + '.jpg', silhouette)
                        image_num += 1
                    cv2.imshow("silhouette", silhouette)

            # draw the hand
            cv2.rectangle(one_frame, (left, top), (right, bottom), (255, 255, 255), 2)

            # increment the number of frames
            num_frames += 1
            cv2.imshow("Data Base Creator", one_frame)


            
            if  image_num > 100:
                # 100 per keyboard press 
                start_recording = False
                image_num = 0
                base += 100
                print("stoped recording")
            if keypress == ord("q"):
                # free up memory
                camera.release()
                cv2.destroyAllWindows()
                break
            if keypress == ord("s"):
                start_recording = True


main()



