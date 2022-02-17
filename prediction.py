from TrackNet import ResNet_Track
from focal_loss import BinaryFocalLoss
import numpy as np
import cv2
import time
from utils import genHeatMap

class ShuttlePredictor:

    def __init__(self):
        self.BATCH_SIZE = 2
        self.HEIGHT = 288 
        self.WIDTH = 512
        self.model = ResNet_Track(input_shape=(3, self.HEIGHT, self.WIDTH))

    def predict_frame(self,img_input):
        WIDTH,HEIGHT = self.WIDTH, self.HEIGHT
        img_input = np.concatenate(img_input, axis=2)
        img_input = cv2.resize(img_input, (WIDTH, HEIGHT))
        img_input = np.moveaxis(img_input, -1, 0)
        img_input = np.expand_dims(img_input, axis=0)
        img_input = img_input.astype('float')/255.
        y_pred = self.model.predict(img_input, batch_size=self.BATCH_SIZE)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype('float32')
