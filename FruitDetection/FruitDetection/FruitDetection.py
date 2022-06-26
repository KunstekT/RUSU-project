import tensorflow as tf
tf.__version__

import sys
sys.path.append("..")

import cv2
import numpy as np
from glob import glob
from models import Yolov4
import matplotlib.pyplot as plt

model = Yolov4(weight_path='yolo-obj_final.weights',
               class_name_path='class_names/fruits_classes.txt')

model.predict("img/grapes3.jpg", random_color=True)
