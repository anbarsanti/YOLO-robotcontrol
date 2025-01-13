"""
﷽
by @anbarsanti
"""
import sys
sys.path.append('../RTDE_Python_Client_Library')
from r2r_functions import *
import numpy as np

# ## ====================== INITIALIZATION OF TRACKING STUFF ==================================
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

## ========================= INITIALIZATION OF ROBOT COMMUNICATION  =========================
# ROBOT_HOST = "10.149.230.168" # in robotics lab
ROBOT_HOST = "192.168.18.14"  # virtual machine in from linux host
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"
FREQUENCY = 1000 # send data in 500 Hz instead of default 125Hz
time_start = time.time()
plotter = True
trajectory_time = 8
# Setpoints to move the robot to
start_pose = [0.4, -0.6, 0, 0, 0, 0]
desired_value = [-0.2, -0.5, 0.2, 0.7, 0.3, -0.1]*5

## =========================  UR5E INITIALIZATION ==================================================
con, state, watchdog, setp = UR5e_init(ROBOT_HOST, ROBOT_PORT, FREQUENCY, config_filename)
time_plot = [0]
actual_p = np.array(state.actual_TCP_pose)
actual_q = np.array(state.actual_q)
q_dot = np.zeros((6,1))

## =========================  UR5E MOVE TO INITIAL POSITION =========================
con, state, watchdog, setp = UR5e_start(con, state, watchdog, setp)

# ## ======================= IMAGE JACOBIAN ==================================
p0 = ([[0.5],[0.3]])
p1 = ([[0.5],[0.9]])
p2 = ([[0.8],[0.7]])
p3 = ([[0.8],[0.3]])
p4 = ([[0.7],[0.1]])
p5 = ([[0.5],[0.1]])
p6 = ([[0.2],[0.1]])
p7 = ([[0.2],[0.3]])
p8 = ([[0.2],[0.7]])
c = [[0.5], [0.5]]
J = image_jacobian = J_image_n(c) @ actual_q

# ## ======================= UR5E STARTS  ==================================


while True:
	# Send the q_dot to UR5e
	list_to_setp(setp, q_dot)
	con.send(setp)
	state = con.receive()
	new_actual_p = np.array(state.actual_TCP_pose)
	new_actual_q = np.array(state.actual_q)
				
# Release resources
cap.release()
cv2.destroyAllWindows()

## =========================  DISCONNECTING THE UR5E ========================================
con.send(watchdog)
con.send_pause()
con.disconnect()

## =========================  FINAL PLOTTING ==================================================
# final_plotting(time_plot, actual_p, actual_q)
