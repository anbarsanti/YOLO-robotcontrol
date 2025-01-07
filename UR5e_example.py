"""
﷽
Alhamdulillah, this code is written based on servoj example from:
1. https://github.com/davizinho5/RTDE_control_example
2. https://github.com/danielstankw/Servoj_RTDE_UR5
by @anbarsanti
"""
import sys

sys.path.append('../RTDE_Python_Client_Library')
import logging
from YOLOv11.rtde import rtde as rtde
from YOLOv11 import rtde as rtde_config
from matplotlib import pyplot as plt
from YOLOv11.min_jerk_planner_translation import PathPlanTranslation
import time
from r2r_functions import *

## ====================== ROBOT COMMUNICATION STUFF ==================================
# ROBOT_HOST = "10.149.230.168" # in robotics lab
ROBOT_HOST = "192.168.18.14" # virtual machine in from linux host
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"
FREQUENCY = 500

logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe("state")
setp_names, setp_types = conf.get_recipe("setp")
watchdog_names, watchdog_types = conf.get_recipe("watchdog")

## ===========================  ESTABLISH CONNECTION ===========================
con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
connection_state = con.connect()
print("Successfully connected to the robot:", connection_state)

# ===========================  GET CONTROLLER CONNECTION ======================
con.get_controller_version()
print("Get controller version:", con.get_controller_version())

# =========================== SETUP RECIPES ================================
# Received the data from the robot
con.send_output_setup(state_names, state_types, FREQUENCY)
# Configure an input package that the external application will send to the robot controller
setp = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)
print("Setup recipes done")

# =========================== INITIALIZATION ================================
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
print("Initialization done")

# Setpoints to move the robot to
start_pose = [0.38895, -0.62563, 0, 0, 0, 0]
desired_pose = [-0.1827681851594755, -0.53539320093064, 0.2077025734923525, 0.6990025901302169, 0.30949715741835195, -0.10065200008528846]
print("Initialization of setpoints to move the robot to, done")


# ====================== START DATA SYNCHRONIZATION ======================
if not con.send_start():
    sys.exit()
    print("system exit")
state = con.receive()
tcp1 = state.actual_TCP_pose
print("actual_TCP_pose:", tcp1)
print("data synchronization done....")

# ============================ MODE 1 = CONNECTION AND EXECUTE MOVEJ ============================
while True:
    print('Please click CONTINUE on the Polyscope') # Boolean 1 is False
    state = con.receive()
    con.send(watchdog)
    if state.output_bit_registers0_to_31 == True:
        print('Robot Program can proceed to mode 1\n') # Boolean 1 is True
        break

print("---------------- Executing Initial Movement (MoveJ)  --------------\n")

watchdog.input_int_register_0 = 1
con.send(watchdog)  # sending mode == 1
list_to_setp(setp, start_pose)  # changing initial pose to setp
con.send(setp) # sending initial pose

while True:
    # print('Waiting for movej() to finish')
    state = con.receive()
    con.send(watchdog)
    if state.output_bit_registers0_to_31 == False:
        print('Initial Movement (MoveJ) done, proceeding to Loop Movement\n')
        break

# ============================ MODE 2 = EXECUTE SPEEDJ/SERVOJ ============================
print("------------------- Executing Loop Movement  -----------\n")
watchdog.input_int_register_0 = 2
con.send(watchdog)  # sending mode == 2

# ============================ CONTROL LOOP INITIALIZATION ============================
trajectory_time = 8  # time of min_jerk trajectory
planner = PathPlanTranslation(start_pose, desired_pose, trajectory_time)
dt = 1/500  # 500 Hz    # frequency
plotter = False
state = con.receive()
tcp = state.actual_TCP_pose
t_current = 0
t_start = time.time()
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
    actual_vx = []
    actual_vy = []
    actual_vz = []

# =================================== CONTROL LOOP  ===================================
while time.time() - t_start < trajectory_time:
    t_init = time.time()
    state = con.receive()
    t_prev = t_current
    t_current = time.time() - t_start

    if state.runtime_state > 1:     # read state from the robot
        if t_current <= trajectory_time:
            [position_ref, velocity_ref, acceleration_ref] = planner.trajectory_planning(t_current)

        # ------------------ Read State for Plotting Purpose -----------------------
        current_p = state.actual_TCP_pose
        current_q = state.actual_q
        current_qd = state.actual_qd

        pose = velocity_ref.tolist() + [0,0,0]

        list_to_setp(setp, pose)
        con.send(setp)

        # ------------------ Write State for Plotting Purpose -----------------------
        if plotter:
            time_plot.append(time.time()-t_start)
            actual_px.append(current_p[0])
            actual_py.append(current_p[1])
            actual_pz.append(current_p[2])
            actual_q1.append(current_q[0])
            actual_q2.append(current_q[1])
            actual_q3.append(current_q[2])
            actual_qd1.append(current_qd[0])
            actual_qd2.append(current_qd[1])
            actual_qd3.append(current_qd[2])

print('--------------------------------------------------------------------------------\n')
print(f"Alhamdulillah, it took {time.time()-t_start}s to execute the movement")
state = con.receive()
print('--------------------------------------------------------------------------------\n')
print("Last Position:", state.actual_TCP_pose)

# =================================== MODE 3  ===================================
watchdog.input_int_register_0 = 3
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
