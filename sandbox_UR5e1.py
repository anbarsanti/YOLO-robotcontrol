"""
﷽
by @anbarsanti
"""
import sys
sys.path.append('../RTDE_Python_Client_Library')
import logging
import rtde as rtde
import rtde as rtde_config
from matplotlib import pyplot as plt
from YOLOv11.min_jerk_planner_translation import PathPlanTranslation
import time
from r2r_functions import *
import numpy as np
import math
import torch


## ========================= INITIALIZATION OF ROBOT COMMUNICATION  =========================
ROBOT_HOST = "10.149.230.168" # in robotics lab
# ROBOT_HOST = "192.168.18.13"  # virtual machine in from linux host
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"
FREQUENCY = 250 # send data in 500 Hz instead of default 125Hz
time_start = time.time()
trajectory_time = 10

## =========================  UR5E INITIALIZATION ==================================================
con, state, watchdog, setp = UR5e_init(ROBOT_HOST, ROBOT_PORT, FREQUENCY, config_filename)

# Initialization of Plotting Variable
area = 0
area_plot = [0]
time_plot = [0]
time_start = time.time()
q_dot = np.zeros((6, 1))
q_dot_plot = np.zeros((6, 1))
epsilon = np.zeros((6, 1))
epsilon_plot = np.zeros((6, 1))
actual_p = np.array(state.actual_TCP_pose)
new_actual_p = np.array(state.actual_TCP_pose)
actual_q = np.array(state.actual_q)
new_actual_q = np.array(state.actual_q)

## =========================  UR5E MOVE TO INITIAL POSITION =========================
con, state, watchdog, setp = UR5e_start(con, state, watchdog, setp)

## ========================= UR5E LOOPING MOVE TO DESIRED VALUE =========================
x0 = [[0.5],[0.3]]
x1 = [[0.5],[0.9]]
x2 = [[0.8],[0.7]]
x3 = [[0.8],[0.3]]
x4 = [[0.7],[0.1]]
x5 = [[0.5],[0.1]]
x6 = [[0.2],[0.1]]
x7 = [[0.2],[0.3]]
x8 = [[0.2],[0.7]]

x_desired = x7 		# in image space
x_actual = x0	# in tools space
# x_actual_imagespace = R_wr3 @ (np.vstack((x_actual, 0)))		# in image space
# delta_x = np.subtract(x_desired, x_actual_imagespace[0:2])
delta_x = np.subtract(x_desired, x_actual)
print("delta_x", delta_x)

p_dot = - R_ri6 @ np.linalg.pinv(J_image_n(x_actual)) @ delta_x # ---> this is the correct way
p_dot[3][0] = 0; p_dot[4][0] = 0; p_dot[5][0] = 0
print("p_dot", p_dot)
new_actual_q = new_actual_q.reshape(6,1)
q_dot = 50* np.linalg.pinv(J_r(new_actual_q)) @ p_dot
q_dot[3][0] = 0; q_dot[4][0] = 0; q_dot[5][0] = 0
print("q_dot", q_dot)

# Send the q_dot to UR5e
while time.time() - time_start < trajectory_time:
	list_to_setp(setp, q_dot)
	con.send(setp)
	state = con.receive()
	new_actual_p = np.array(state.actual_TCP_pose)  # dimension (1,6)
	new_actual_q = np.array(state.actual_q)  # dimension (1,6)

	## =================== SAVE FOR PLOTTING AND ANALYSIS ===================================
	time_plot.append(time.time() - time_start)
	area_plot.append(area)
	epsilon_plot = np.append(epsilon_plot, epsilon, axis=1)
	actual_p = np.vstack((actual_p, new_actual_p))
	actual_q = np.vstack((actual_q, new_actual_q))
	q_dot_plot = np.append(q_dot_plot, q_dot, axis=1)
	
## =========================  DISCONNECTING THE UR5E ========================================
con.send(watchdog)
con.send_pause()
con.disconnect()

## =========================  FINAL PLOTTING ==================================================
final_plotting (time_plot, actual_p, actual_q, q_dot_plot, area_plot, epsilon_plot)