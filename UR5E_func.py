import sys

sys.path.append('../RTDE_Python_Client_Library')
import logging
import rtde as rtde
import rtde as rtde_config
from matplotlib import pyplot as plt
from YOLOv11.min_jerk_planner_translation import PathPlanTranslation
import time
from r2r_functions import *

def UR5e_init(ROBOT_HOST, ROBOT_PORT, FREQUENCY, config_filename):
	'''
	Robot communication initialization
	Args: ROBOT_HOST, ROBOT_PORT, config_filename
	Return: con, state, watchdog
	'''
	logging.getLogger().setLevel(logging.INFO)
	
	conf = rtde_config.ConfigFile(config_filename)
	state_names, state_types = conf.get_recipe("state")
	setp_names, setp_types = conf.get_recipe("setp")
	watchdog_names, watchdog_types = conf.get_recipe("watchdog")

	# Establish connection
	con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
	connection_state = con.connect()
	con.get_controller_version()  # Get controller version

	# Received the data from the robot
	con.send_output_setup(state_names, state_types, FREQUENCY)
	# Configure an input package that the external application will send to the robot controller
	setp = con.send_input_setup(setp_names, setp_types)
	watchdog = con.send_input_setup(watchdog_names, watchdog_types)

	# Start data synchronization
	if not con.send_start():
		sys.exit()
		print("system exit")
	state = con.receive()
	
	# Initialization Robot Parameters
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
	
	print("Initialization of Robot Communication Stuff, Alhamdulillah done.")
	
	return con, state, watchdog, setp

def UR5e_start(con, state, watchdog, setp):
	# Execute MoveJ to the start joint position (check polyscope for the init joint)
	while True:
		print('Please click CONTINUE on the Polyscope')  # Boolean 1 is False
		state = con.receive()
		con.send(watchdog)
		if state.output_bit_registers0_to_31 == True:
			print('Robot Program can proceed to loop mode.\n')  # Boolean 1 is True
			break
	
	return con, state, watchdog, setp


# Initialization Robot Communication Parameter
# ROBOT_HOST = "10.149.230.168" # in robotics lab
ROBOT_HOST = "192.168.18.14"  # virtual machine in from linux host
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"
FREQUENCY = 250 # send data in 500 Hz instead of default 125Hz
time_start = time.time()
plotter = False
trajectory_time = 8
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
while True:
	desired_pose = desired_pose*1.001
	list_to_setp(setp, desired_pose)
	con.send(setp)
	# print("moving")
	
	# if plotter:
	# 	# Plotting Purpose
	# 	state = con.receive()
	# 	actual_q = state.actual_q
	# 	actual_qd = state.actual_qd
	# 	actual_p = state.actual_TCP_pose
	#
	# 	time_plot.append(time.time() - time_start)
	# 	actual_px.append(actual_p[0])
	# 	actual_py.append(actual_p[1])
	# 	actual_pz.append(actual_p[2])
	# 	actual_q1.append(actual_q[0])
	# 	actual_q2.append(actual_q[1])
	# 	actual_q3.append(actual_q[2])
	# 	actual_qd1.append(actual_qd[0])
	# 	actual_qd2.append(actual_qd[1])
	# 	actual_qd3.append(actual_qd[2])
	# 	print("moving and recording data....")

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
