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
ROBOT_HOST = "192.168.18.13"  # virtual machine in from linux host
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
actual_q = np.array(state.actual_q)

## =========================  UR5E MOVE TO INITIAL POSITION =========================
con, state, watchdog, setp = UR5e_start(con, state, watchdog, setp)

# ## ======================= IMAGE and UR5E JACOBIAN TEST ==================================
x0 = [[0.5],[0.3]]
x1 = [[0.5],[0.9]]
x2 = [[0.8],[0.7]]
x3 = [[0.8],[0.3]]
x4 = [[0.7],[0.1]]
x5 = [[0.5],[0.1]]
x6 = [[0.2],[0.1]]
x7 = [[0.2],[0.3]]
x8 = [[0.2],[0.7]]
c = x0
k = 1000
delta_x = np.subtract(x1, x0)
p_dot = - R_rc @ (np.linalg.pinv(J_image_n(c)) @ delta_x)
q_dot = k * np.linalg.pinv(J_r(actual_p)) @ p_dot
print("q_dot", q_dot)

# ## ======================= UR5E STARTS  ==================================

while time.time() - time_start < 60:
	# Send the q_dot to UR5e
	list_to_setp(setp, q_dot)
	con.send(setp)
	state = con.receive()
	new_actual_p = np.array(state.actual_TCP_pose)
	new_actual_q = np.array(state.actual_q)
	print("new_actual_q", new_actual_q)
	
	## =================== SAVE FOR PLOTTING AND ANALYSIS ===================================
	time_plot.append(time.time() - time_start)
	area_plot.append(area)
	epsilon_plot = np.append(epsilon_plot, epsilon, axis=1)
	actual_p = np.vstack((actual_p, new_actual_p))
	actual_q = np.vstack((actual_q, new_actual_q))
	q_dot_plot = np.append(q_dot_plot, epsilon, axis=1)
	


## =========================  DISCONNECTING THE UR5E ========================================
con.send(watchdog)
con.send_pause()
con.disconnect()

## =========================  FINAL PLOTTING ==================================================
final_plotting (time_plot, actual_p, actual_q, q_dot_plot, area_plot, epsilon_plot)