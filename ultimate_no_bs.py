import matplotlib.pyplot as plt
from tensorflow import keras
from collections import deque
from focal_loss import BinaryFocalLoss
from TrackNet import ResNet_Track
from parser_1 import parser
from glob import glob
from utils import *
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import getopt
import math
import time
import sys
import cv2
import os
from cgi import test
from asyncore import read
print("[!] Imports completed successfully.")

print("[!] Imports completed successfully.")
######################### GLOBALS ############################
# GLOBAL VALS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
args = parser.parse_args()
tol = args.tol
mag = args.mag
sigma = args.sigma
HEIGHT = args.HEIGHT
WIDTH = args.WIDTH
BATCH_SIZE = 2
FRAME_STACK = args.frame_stack
load_weights = args.load_weights
#video_path = args.video_path
#csv_path = args.label_path
count = 0
opt = keras.optimizers.Adadelta(learning_rate=1.0)
model = ResNet_Track(input_shape=(3, HEIGHT, WIDTH))
model.compile(loss=BinaryFocalLoss(gamma=2), optimizer=opt,
              metrics=[keras.metrics.BinaryAccuracy()])
n_frames = 4000
np_roi_cords = []
roi_cords = []
roi_taken = False
POSITION = None
mode = args.mode
##########################################################
# Load Model in memory
print("[!] Loading weights in memory")
try:
    model.load_weights(load_weights)
    print("Load weights successfully")
except:
    print("Fail to load weights, please modify path in parser.py --load_weights")
# #####################################################################


def draw_roi(image, cords):
    '''
            Draw Roi polylines on frame
    '''
    global POSITION
    color = (255, 0, 0)
    pt = cords
    #np_roi_cords = np.array(roi_cords, np.int32)
    for i in range(len(np_roi_cords)):
        cv2.polylines(image, [np_roi_cords[i]], True, color, 3)
    if cords == (0, 0):
        return image
    # Check if in or out using point polygontest
    in_roi = False  # Initialize as false
    for i in range(len(np_roi_cords)):
        dist = cv2.pointPolygonTest(np_roi_cords[i], pt, False)
        if dist in [1.0, 0.0]:
            in_roi = True

    if POSITION == None:
        POSITION = ((int)(image.shape[1] / 2 - 268 / 2),
                    (int)(image.shape[0] / 2 - 36 / 2))
    if in_roi:
        cv2.putText(image, "INSIDE", POSITION,
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 247, 0), 3)
        return image
    cv2.putText(image, "OUTSIDE", POSITION,
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 247, 0), 3)
    return image


#################################################################
# Take Roi from user.

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
        if len(roi_cords) >= 4:
            cv2.setMouseCallback('roi_window', lambda *args: None)
        roi_cords.append([x, y])


def take_roi(image):
    from EasyROI import EasyROI
    import cv2
    roi_helper = EasyROI()
    global roi_cords,np_roi_cords
    polygon_roi = roi_helper.draw_polygon(frame, 3)
    cropped_polys = roi_helper.crop_roi(frame, polygon_roi)

    rois = polygon_roi.get('roi')
    for roi in rois:
        cords = rois.get(roi).get('vertices')
        roi_cords.append(cords)
        np_roi_cords.append(np.array(cords, np.int32))


print(f"[+] Starting webcam on {mode}")
roi_cap = cv2.VideoCapture(mode)  # for go pro use 1 for webcam use 0
roi_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
roi_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
roi_cap.set(cv2.CAP_PROP_FPS, 30)

# Check if the webcam is opened correctly
if not roi_cap.isOpened():
    raise IOError("Cannot open webcam")
print(f"[+] Started WebCam.")
while roi_cap.isOpened():
    print(f"[!] New Video Frame opened ")
    # Read video capture
    ret, frame = roi_cap.read()
    # Display each frame
    cv2.imshow("FRAME SELECTION", frame)
    # show one frame at a time
    key = cv2.waitKey(0)
    while key not in [ord('q'), ord('k')]:
        key = cv2.waitKey(0)
    # Quit when 'q' is pressed
    if key == ord('q'):
        roi_cap.release()
        break
    cv2.destroyAllWindows()
    if not roi_taken:
        take_roi(frame)
        roi_taken = True
    frame = draw_roi(frame, (0, 0))
    cv2.namedWindow('roi_preview')
    cv2.imshow('roi_preview', frame)
    cv2.waitKey(0)
    if len(roi_cords) > 4:
        print("[!] 4 Cords taken")
        break

cv2.destroyAllWindows()
# #######################################################
# Now ROI is taken, Start Prediction

cap = cv2.VideoCapture(mode)  # for go pro use 1 for webcam use 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

input("[+] Press any key to start processing video")

gray_imgs = deque()
success, image = cap.read()
ratio = image.shape[0] / HEIGHT

size = (int(WIDTH*ratio), int(HEIGHT*ratio))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test123'+'_predict.mp4', fourcc, 30, size)
out.write(image)

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = np.expand_dims(img, axis=2)
gray_imgs.append(img)
for _ in range(FRAME_STACK-1):
    success, image = cap.read()
    out.write(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=2)
    gray_imgs.append(img)
frame_no = FRAME_STACK-1
time_list = []
TP = TN = FP1 = FP2 = FN = 0

while success:
    img_input = np.concatenate(gray_imgs, axis=2)
    img_input = cv2.resize(img_input, (WIDTH, HEIGHT))
    img_input = np.moveaxis(img_input, -1, 0)
    img_input = np.expand_dims(img_input, axis=0)
    img_input = img_input.astype('float')/255.0

    start = time.time()
    y_pred = model.predict(img_input, batch_size=BATCH_SIZE)
    end = time.time()
    time_list.append(end-start)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype('float32')

    h_pred = y_pred[0]*255
    h_pred = h_pred.astype('uint8')

    if np.amax(h_pred) <= 0:
        image = draw_roi(image, (0, 0))
    else:
        cnts, _ = cv2.findContours(
            h_pred[0].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]
        max_area_idx = 0
        max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
        for i in range(1, len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        target = rects[max_area_idx]
        (cx_pred, cy_pred) = (
            int(ratio*(target[0] + target[2] / 2)), int(ratio*(target[1] + target[3] / 2)))
        image = draw_roi(image, (cx_pred, cy_pred))
        cv2.circle(image, (cx_pred, cy_pred), 30, (0, 0, 255), 10)
        cv2.putText(image, "Shuttlecock", (cx_pred, cy_pred),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
    out.write(image)
    cv2.imshow("Predict Preview", image)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()

    success, image = cap.read()
    if success:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)
        gray_imgs.append(img)
        gray_imgs.popleft()
        frame_no += 1
        #cv2.imshow("Preview", img)
        # cv2.waitKey(1)
    count += 1

out.release()
total_time = sum(time_list)
cap.release()
cv2.destroyAllWindows()