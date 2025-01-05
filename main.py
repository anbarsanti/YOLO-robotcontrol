"""
﷽
by @anbarsanti
"""

import numpy as np
import math
import torch
from r2r_functions import *
import sys

## ================================== INITIALIZATION OF TRACKING STUFF ==================================
OBB = False
model = YOLO("model/yolo11-hbb-toy-12-01.pt") # toys for HBB object tracking
# model = YOLO("model/yolo11-obb-11-16-watercan.pt") # watercan for OBB object tracking
# model = YOLO("model/yolo11n.pt") # object tracking with HBB

if OBB==True: # Initialization for OBB case
	desired_box = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	reaching_box = [0, 0, 0, 0, 0, 0, 0, 0, 0]
else: # Initialization for HBB case
	desired_box = [0, 0, 0, 0, 0]
	reaching_box = [0, 0, 0, 0, 0]

## ================================== TRACKING STARTS ==================================
for detected_box in track_from_webcam(model, OBB=OBB):

	# For HBB case
	if detected_box[0] == 1: # Detecting Toy's Box
		desired_box = detected_box
	else:
		reaching_box = detected_box

	print("reaching_box", reaching_box)
	print("desired_box", desired_box)
	print("Area of intersection:", intersection_area_HBB(desired_box, reaching_box))

	q_dot = r2r_control(reaching_box, desired_box, actual_q, OBB=OBB)
	print("q_dot:", q_dot)




