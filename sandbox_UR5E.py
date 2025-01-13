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
ROBOT_HOST = "192.168.18.3"  # virtual machine in from linux host
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
area_plot = [0]
time_plot = [0]
time_start = time.time()
q_dot = np.zeros((6, 1))
q_dot_plot = np.empty((6, 1))
epsilon = np.empty((6, 1))
epsilon_plot = np.empty((6, 1))
actual_p = np.array(state.actual_TCP_pose)
actual_q = np.array(state.actual_q)

## =========================  UR5E MOVE TO INITIAL POSITION =========================
con, state, watchdog, setp = UR5e_start(con, state, watchdog, setp)

# ## ======================= UR5E JACOBIAN TEST ==================================
p_dot0 = ([0, 0, 0, 0, 0, 0])
p_dot1 = ([100, 0, 0, 0, 0, 0])
delta_p = np.subtract(p_dot1, p_dot0)
q_dot = np.linalg.pinv(J_r(p_dot0)) @ delta_p
print("q_dot", q_dot)

# ## ======================= UR5E STARTS  ==================================

while True:
	# Send the q_dot to UR5e
	list_to_setp(setp, q_dot)
	con.send(setp)
	state = con.receive()
	new_actual_p = np.array(state.actual_TCP_pose)
	new_actual_q = np.array(state.actual_q)
	print("new_actual_p", new_actual_p)
	
	## =================== SAVE FOR PLOTTING AND ANALYSIS ===================================
	time_plot.append(time.time() - time_start)
	area_plot.append(area)
	epsilon_plot = np.append(epsilon_plot, epsilon, axis=1)
	actual_p = np.vstack((actual_p, new_actual_p))
	actual_q = np.vstack((actual_q, new_actual_q))
	q_dot_plot = np.append(q_dot_plot, q_dot, axis=1)
				
# Release resources
cap.release()
cv2.destroyAllWindows()

## =========================  DISCONNECTING THE UR5E ========================================
con.send(watchdog)
con.send_pause()
con.disconnect()

## =========================  FINAL PLOTTING ==================================================
final_plotting (time_plot, actual_p, actual_q, q_dot_plot, area_plot, epsilon_plot)