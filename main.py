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
import rtde as rtde
import rtde as rtde_config
from matplotlib import pyplot as plt
from min_jerk_planner_translation import PathPlanTranslation
import time


# ## ================================== INITIALIZATION OF TRACKING STUFF ==================================
# OBB = False
# model = YOLO("model/yolo11-hbb-toy-12-01.pt") # toys for HBB object tracking
# # model = YOLO("model/yolo11-obb-11-16-watercan.pt") # watercan for OBB object tracking
# # model = YOLO("model/yolo11n.pt") # object tracking with HBB
#
# if OBB==True: # Initialization for OBB case
# 	desired_box = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# 	reaching_box = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# else: # Initialization for HBB case
# 	desired_box = [0, 0, 0, 0, 0]
# 	reaching_box = [0, 0, 0, 0, 0]
#
# ## ================================== TRACKING STARTS ==================================
# for detected_box in track_from_webcam(model, OBB=OBB):
#
# 	# For HBB case
# 	if detected_box[0] == 1: # Detecting Toy's Box
# 		desired_box = detected_box
# 	else:
# 		reaching_box = detected_box
#
# 	print("reaching_box", reaching_box)
# 	print("desired_box", desired_box)
# 	print("Area of intersection:", intersection_area_HBB(desired_box, reaching_box))
#
# 	q_dot = r2r_control(reaching_box, desired_box, actual_q, OBB=OBB)
# 	print("q_dot:", q_dot)


# Initialization Robot Communication Parameter
# ROBOT_HOST = "10.149.230.168" # in robotics lab
ROBOT_HOST = "192.168.18.14"  # virtual machine in from linux host
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"
FREQUENCY = 250  # send data in 500 Hz instead of default 125Hz
time_start = time.time()
plotter = False
t = 8
if plotter:
	time_plot = []
	actual_q1 = []
	actual_q2 = []
	actual_q3 = []
	actual_qd1 = []
	actual_qd2 = []
	actual_qd3 = []
	actual_px = []
	actual_py = []
	actual_pz = []

# UR5e Init
con, state, watchdog, setp = UR5e_init(ROBOT_HOST, ROBOT_PORT, FREQUENCY, config_filename)

# UR5e Move to Initial position
con, state, watchdog, setp = UR5e_start(con, state, watchdog, setp)

# Setpoints to move the robot to
start_pose = [0.4, -0.6, 0, 0, 0, 0]
desired_pose = [-0.1827681851594755, -0.53539320093064, 0.2077025734923525, 0.6990025901302169, 0.30949715741835195,
					 -0.10065200008528846]

# Execute moving loop for the period of trajectory_time
# con, state, watchdog, setp = UR5e_loop(con, state, watchdog, setp, desired_pose, plotter)
while True:
	list_to_setp(setp, desired_pose)
	con.send(setp)
	print("moving")

con.send(watchdog)
con.send_pause()
con.disconnect()

# =================================== PLOTTING  ===================================
if plotter:
	# ----------- Tool Position -------------
	plt.figure()
	plt.plot(time_plot, actual_px, label="Actual Tool Position in x[m]")
	plt.legend()
	plt.grid()
	plt.ylabel('Tool Position in x[m]')
	plt.xlabel('Time [sec]')
	
	plt.figure()
	plt.plot(time_plot, actual_py, label="Actual Tool Position in y[m]")
	plt.legend()
	plt.grid()
	plt.ylabel('Tool Position in y[m]')
	plt.xlabel('Time [sec]')
	
	plt.figure()
	plt.plot(time_plot, actual_px, label="Actual Tool Position in z[m]")
	plt.legend()
	plt.grid()
	plt.ylabel('Tool Position in z[m]')
	plt.xlabel('Time [sec]')
	
	# ----------- Joint Position -------------
	plt.figure()
	plt.plot(time_plot, actual_q1, label="Actual Joint Position in 1st Joint")
	plt.legend()
	plt.grid()
	plt.ylabel('Joint Position in x')
	plt.xlabel('Time [sec]')
	
	plt.figure()
	plt.plot(time_plot, actual_q2, label="Actual Joint Position in 2nd Joint")
	plt.legend()
	plt.grid()
	plt.ylabel('Joint Position in y')
	plt.xlabel('Time [sec]')
	
	plt.figure()
	plt.plot(time_plot, actual_q3, label="Actual Joint Position in 3rd Joint")
	plt.legend()
	plt.grid()
	plt.ylabel('Joint Position in z')
	plt.xlabel('Time [sec]')
	
	# ----------- Desired Joint -------------
	plt.figure()
	plt.plot(time_plot, actual_qd1, label="Actual Joint Velocity in 1st Joint")
	plt.legend()
	plt.grid()
	plt.ylabel('Joint Velocity in x')
	plt.xlabel('Time [sec]')
	
	plt.figure()
	plt.plot(time_plot, actual_qd2, label="Actual Joint Velocity in 2nd Joint")
	plt.legend()
	plt.grid()
	plt.ylabel('Joint Velocity in y')
	plt.xlabel('Time [sec]')
	
	plt.figure()
	plt.plot(time_plot, actual_qd3, label="Actual Joint Velocity in 3rd Joint")
	plt.legend()
	plt.grid()
	plt.ylabel('Joint Velocity in z')
	plt.xlabel('Time [sec]')
	plt.show()
