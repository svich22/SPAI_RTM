import os
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
import csv
import sys
import pandas as pd

try:
	input_video_path = sys.argv[1]
	input_csv_path = sys.argv[2]
	#output_video_path = sys.argv[3]
	if (not input_video_path) or (not input_csv_path):
		raise ''
	if not os.path.exists(input_csv_path):
		raise ''
except:
	print('usage: python3 show_trajectory.py <input_video_path> <input_csv_path>')
	exit(1)

df = pd.read_csv(input_csv_path)

def get_csv_info(no):
	''' get info of csv '''
	row = df.iloc[no]
	d = {
		'frame':row['Frame'],
		'visibility':bool(row['Visibility']),
		'x':row['X'],
		'y':row['Y']
		#'time':row['Time'],
	}
	return d

video = cv2.VideoCapture(input_video_path)
frame_cnt = 0
trajectory_limit = 8
dets = []

while True:
	ret, img = video.read()
	print(img.shape)
	pred_info = get_csv_info(frame_cnt)
	print("Frame cnt ", frame_cnt)
	visibility = pred_info.get('visibility')
	if frame_cnt < trajectory_limit:
		if visibility:
			dets.append([round(pred_info['x']),round(pred_info['y'])])
	else:
		# Perform FIFO Queue
		if visibility:
			dets.pop(0)
			dets.append([round(pred_info['x']),round(pred_info['y'])])
	frame_cnt += 1
	if len(dets) <= 1:
		continue
	new_img = cv2.polylines(img, [np.array(dets)], 1, (0,255,255))
	cv2.namedWindow('output', cv2.WINDOW_NORMAL)
	cv2.imshow('output',new_img)
	cv2.waitKey(1)
