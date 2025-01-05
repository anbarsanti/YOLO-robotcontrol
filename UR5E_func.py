import sys

sys.path.append('../RTDE_Python_Client_Library')
import logging
from YOLOv11.rtde import rtde as rtde
from YOLOv11 import rtde as rtde_config
from matplotlib import pyplot as plt
from YOLOv11.min_jerk_planner_translation import PathPlanTranslation
import time
from r2r_functions import *

# Define Function

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

	print("Initialization of Robot Communication Stuff, Alhamdulillah done.")
	
	return con, state, watchdog, setp

# def UR5e_loop()

# Initialization Robot Communication Parameter
# ROBOT_HOST = "10.149.230.168" # in robotics lab
ROBOT_HOST = "192.168.18.14"  # virtual machine in from linux host
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"
FREQUENCY = 500 # send data in 500 Hz instead of default 125Hz

# Start UR5e Init
con, state, watchdog, setp = UR5e_init(ROBOT_HOST, ROBOT_PORT, FREQUENCY, config_filename)

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

# Setpoints to move the robot to
start_pose = [0.4, -0.6, 0, 0, 0, 0]
desired_pose = [-0.1827681851594755, -0.53539320093064, 0.2077025734923525, 0.6990025901302169, 0.30949715741835195,
					 -0.10065200008528846]

actual_q = state.actual_q
actual_qd = state.actual_qd
actual_tcp = state.actual_TCP_pose
print("actual_q:", actual_q)
print("actual_qd:", actual_qd)
print("actual_tcp:", actual_tcp)

# Execute MoveJ to the start joint position (check polyscope for the init joint)
while True:
	print('Please click CONTINUE on the Polyscope')  # Boolean 1 is False
	state = con.receive()
	con.send(watchdog)
	if state.output_bit_registers0_to_31 == True:
		print('Robot Program can proceed to loop mode.\n')  # Boolean 1 is True
		break

# ============================ MODE 2 = EXECUTE SPEEDJ/SERVOJ ============================
print("------------------- Executing Loop Movement  -----------\n")
watchdog.input_int_register_0 = 2
con.send(watchdog)  # sending mode == 2

# ============================ CONTROL LOOP INITIALIZATION ============================
trajectory_time = 8  # time of min_jerk trajectory
planner = PathPlanTranslation(start_pose, desired_pose, trajectory_time)
dt = 1 / 500  # 500 Hz    # frequency
state = con.receive()
tcp = state.actual_TCP_pose
t_current = 0
t_start = time.time()

# =================================== CONTROL LOOP  ===================================
while time.time() - t_start < trajectory_time:
	t_init = time.time()
	state = con.receive()
	t_prev = t_current
	t_current = time.time() - t_start
	
	if state.runtime_state > 1:  # read state from the robot
		if t_current <= trajectory_time:
			[position_ref, velocity_ref, acceleration_ref] = planner.trajectory_planning(t_current)
		
		# ------------------ Read State for Plotting Purpose -----------------------
		current_p = state.actual_TCP_pose
		current_q = state.actual_q
		current_qd = state.actual_qd
		
		pose = velocity_ref.tolist() + [0, 0, 0]
		
		list_to_setp(setp, pose)
		con.send(setp)
		

print('--------------------------------------------------------------------------------\n')
print(f"Alhamdulillah, it took {time.time() - t_start}s to execute the movement")
state = con.receive()
print('--------------------------------------------------------------------------------\n')
print("Last Position:", state.actual_TCP_pose)

# =================================== MODE 3  ===================================
watchdog.input_int_register_0 = 3
con.send(watchdog)
con.send_pause()
con.disconnect()
