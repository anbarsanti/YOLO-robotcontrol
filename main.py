"""
﷽
by @anbarsanti
"""

import numpy as np
import math
import torch
from r2r_functions import *
import sys

sys.path.append('../RTDE_Python_Client_Library')
import logging
from YOLOv11.rtde import rtde as rtde
from YOLOv11 import rtde as rtde_config
from matplotlib import pyplot as plt
from YOLOv11.min_jerk_planner_translation import PathPlanTranslation
import time

## ================================== INITIALIZATION OF ROBOT COMMUNICATION STUFF ==================================
# ROBOT_HOST = "10.149.230.168" # in robotics lab
ROBOT_HOST = "192.168.18.12" # virtual machine in from linux host
ROBOT_PORT = 30004
FREQUENCY = 500 # send data in 500 Hz instead of default 125Hz
config_filename = "control_loop_configuration.xml"

logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe("state")
setp_names, setp_types = conf.get_recipe("setp")
watchdog_names, watchdog_types = conf.get_recipe("watchdog")

setp.input_double_register_0 = 0
setp.input_double_register_1 = 0
setp.input_double_register_2 = 0
setp.input_double_register_3 = 0
setp.input_double_register_4 = 0
setp.input_double_register_5 = 0
setp.input_bit_registers0_to_31 = 0


# The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
watchdog.input_double_register_0 = 0
watchdog.input_int_register_0 = 0
print("Initialization of Robot Communication Stuff, Alhamdulillah done")

# ===========================  ESTABLISH ROBOT CONNECTION ===========================
con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
connection_state = con.connect()
print("Successfully connected to the robot:", connection_state)
con.get_controller_version() # Get controller version

# Received the data from the robot
con.send_output_setup(state_names, state_types, FREQUENCY)
# Configure an input package that the external application will send to the robot controller
setp = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)
print("Setup recipes done")

# ====================== START ROBOT DATA SYNCHRONIZATION ======================
if not con.send_start():
    sys.exit()
    print("system exit")
state = con.receive()
tcp1 = state.actual_TCP_pose
print("actual_TCP_pose:", tcp1)
print("data synchronization done....")

## ================================== INITIALIZATION OF TRACKING STUFF ==================================
OBB = False
model = YOLO("model/yolo11-hbb-toy-12-01.pt") # toys for HBB object tracking
# model = YOLO("model/yolo11-obb-11-16-watercan.pt") # watercan for OBB object tracking
# model = YOLO("model/yolo11n.pt") # object tracking with HBB

actual_q =

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
	
	