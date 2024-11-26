import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

# Initialize RTDE
ROBOT_IP = "10.149.230.20"
ROBOT_PORT = 30004
config_filename = "/home/anbarsanti/Dropbox/YOLOv11/control/control.xml"

con = rtde.RTDE(ROBOT_IP, ROBOT_PORT)
con.connect()

# Load the configuration file
config = rtde_config.ConfigFile(config_filename)
state_names, state_types = config.get_recipe("state")
setp_names, setp_types = config.get_recipe("setp")

# Initialize input and output
setp = con.send_input_setup(setp_names, setp_types)
state = con.send_output_setup(state_names, state_types)

# Start data synchronization
if not con.send_start():
    print("Failed to start data synchronization")
    exit()

# Define the desired position (X, Y, Z) in meters and orientation (Rx, Ry, Rz) in radians
desired_pose = [-0.143, -0.435, 0.20, -0.001, 3.12, 0.04]

# Set the target pose
setp.input_double_register_0 = desired_pose[0]
setp.input_double_register_1 = desired_pose[1]
setp.input_double_register_2 = desired_pose[2]
setp.input_double_register_3 = desired_pose[3]
setp.input_double_register_4 = desired_pose[4]
setp.input_double_register_5 = desired_pose[5]

# Send the command to the robot
con.send(setp)

# Wait for the robot to reach the target position
while True:
    state = con.receive()
    if state is not None:
        if state.output_int_register_0 == 1:
            print("Target position reached")
            break

# Disconnect from the robot
con.disconnect()