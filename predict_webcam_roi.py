import os
import cv2
import sys
import time
import math
import getopt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from utils import *
from glob import glob
from parser_1 import parser
from TrackNet import ResNet_Track
from focal_loss import BinaryFocalLoss
from collections import deque
from tensorflow import keras
import os


def draw_roi(img):
    color = (255, 0, 0)
    cv2.polylines(img, [np_roi_cords], True, color, 3)
    return img


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
        if len(roi_cords) >= 4:
            cv2.setMouseCallback('image', lambda *args: None)
        roi_cords.append([x, y])


def draw_inside(img, inside):
    if inside:
        cv2.putText(img, "INSIDE", position,
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 247, 0), 3)
        return img
    # cv2.putText(img,"OUTSIDE",position,cv2.FONT_HERSHEY_SIMPLEX,3,(255, 247, 0),3)
    return img


# GLOBAL VALS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
args = parser.parse_args()
tol = args.tol
mag = args.mag
sigma = args.sigma
HEIGHT = args.HEIGHT
WIDTH = args.WIDTH
BATCH_SIZE = 4
FRAME_STACK = args.frame_stack
load_weights = args.load_weights
#video_path = args.video_path
csv_path = args.label_path
count = 0
opt = keras.optimizers.Adadelta(learning_rate=1.0)
model = ResNet_Track(input_shape=(3, HEIGHT, WIDTH))
model.compile(loss=BinaryFocalLoss(gamma=2), optimizer=opt,
              metrics=[keras.metrics.BinaryAccuracy()])
n_frames = 5000
np_roi_cords = []
roi_cords = []


try:
    model.load_weights(load_weights)
    print("Load weights successfully")
except:
    print("Fail to load weights, please modify path in parser.py --load_weights")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # for go pro use 1 for webcam use 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

fps = int(cap.get(cv2.CAP_PROP_FPS))
video_name = "test123.mp4"

if not os.path.isfile(csv_path) and not csv_path.endswith('.csv'):
    compute = False
    info = {
        idx: {
            'Frame': idx,
            'Ball': 0,
            'x': -1,
            'y': -1
        } for idx in range(n_frames)
    }
    print("Predict only, will not calculate accurracy")
else:
    compute = True
    info = load_info(csv_path)
    if len(info) != n_frames:
        print("Number of frames in video and dictionary are not the same!")
        print("Fail to load, predict only.")
        compute = False
        info = {
            idx: {
                'Frame': idx,
                'Ball': 0,
                'x': -1,
                'y': -1
            } for idx in range(n_frames)
        }
    else:
        print("Load csv file successfully")


print('Beginning predicting......')


# img_input initialization
gray_imgs = deque()
success, image = cap.read()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', image)
position = ((int)(image.shape[1] / 2 - 268 / 2),
            (int)(image.shape[0] / 2 - 36 / 2))
for i in range(0, 4):
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    if len(roi_cords) >= 4:
        cv2.setMouseCallback('image', lambda *args: None)


f = open('test123' + '_predict.csv', 'w')
f.write('Frame,Ball,X,Y\n')

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
    img_input = img_input.astype('float')/255.

    start = time.time()
    y_pred = model.predict(img_input, batch_size=BATCH_SIZE)
    end = time.time()
    time_list.append(end-start)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype('float32')
    y_true = []
    if info[frame_no]['Ball'] == 0:
        y_true.append(genHeatMap(WIDTH, HEIGHT, -1, -1, sigma, mag))
    else:
        y_true.append(genHeatMap(WIDTH, HEIGHT, int(
            info[frame_no]['x']/ratio), int(info[frame_no]['y']/ratio), sigma, mag))

    tp, tn, fp1, fp2, fn = confusion(y_pred, y_true, tol)
    TP += tp
    TN += tn
    FP1 += fp1
    FP2 += fp2
    FN += fn

    h_pred = y_pred[0]*255
    h_pred = h_pred.astype('uint8')

    if np.amax(h_pred) <= 0:
        f.write(str(count)+',0,-1,-1,'+'\n')
        image = np.copy(image)
        draw_roi(image)
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

        f.write(str(count)+',1,'+str(cx_pred)+','+str(cy_pred)+'\n')
        image = np.copy(image)
        draw_roi(image)
        cv2.circle(image, (cx_pred, cy_pred), 30, (0, 0, 255), 10)
        cv2.putText(image, "Shuttlecock", (cx_pred, cy_pred),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
    out.write(image)
    cv2.imshow("Predict Preview", image)
    cv2.waitKey(1)

    # DK WHAT IT DOES `\_(^^)_/`
    # success, image = cap.read()
    # if success:
    # 	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 	img = np.expand_dims(img, axis=2)
    # 	gray_imgs.append(img)
    # 	gray_imgs.popleft()
    # 	frame_no += 1
    # 	#cv2.imshow("Preview", img)
    # 	#cv2.waitKey(1)
    # count += 1


f.close()
out.release()
total_time = sum(time_list)
cap.release()
cv2.destroyAllWindows()

if compute:
    print('==========================================================')
    accuracy, precision, recall = compute_acc((TP, TN, FP1, FP2, FN))
    avg_acc = (accuracy + precision + recall)/3

    print("Number of true positive:", TP)
    print("Number of true negative:", TN)
    print("Number of false positive FP1:", FP1)
    print("Number of false positive FP2:", FP2)
    print("Number of false negative:", FN)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Total Time:", total_time)
    print('(ACC + Pre + Rec)/3:', avg_acc)

print('Done......')
