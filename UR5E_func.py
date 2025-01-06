"""
﷽
by @anbarsanti
"""

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
import sys

## ========================= INITIALIZATION OF ROBOT COMMUNICATION  =========================
# ROBOT_HOST = "10.149.230.168" # in robotics lab
ROBOT_HOST = "192.168.18.14"  # virtual machine in from linux host
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"
FREQUENCY = 250 # send data in 500 Hz instead of default 125Hz
time_start = time.time()
plotter = True
trajectory_time = 8
# Setpoints to move the robot to
start_pose = [0.4, -0.6, 0, 0, 0, 0]
desired_value = [-0.2, -0.5, 0.2, 0.7, 0.3, -0.1]*5

## =========================  UR5E INITIALIZATION ==================================================
con, state, watchdog, setp = UR5e_init(ROBOT_HOST, ROBOT_PORT, FREQUENCY, config_filename)

## =========================  UR5E MOVE TO INITIAL POSITION =========================
con, state, watchdog, setp = UR5e_start(con, state, watchdog, setp)

## ========================= UR5E LOOPING MOVE TO DESIRED VALUE =========================
time_plot = [0]
actual_p = np.array(state.actual_TCP_pose)
actual_q = np.array(state.actual_q)
actual_qd = np.array(state.actual_qd)
con, state, watchdog, setp, actual_p, actual_q, actual_qd= UR5e_loop(con, state, watchdog, setp, desired_value,
																							time_start, trajectory_time, time_plot,
																							actual_p, actual_q, actual_qd)
## =========================  DISCONNECTING THE UR5E ========================================
con.send(watchdog)
con.send_pause()
con.disconnect()

## =========================  FINAL PLOTTING ==================================================
final_plotting(time_plot, actual_p, actual_q, actual_qd)