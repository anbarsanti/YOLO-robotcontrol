import socket
import struct
import math
import numpy as np
import time
import re
from PIL import Image
from numpy import array as matrix, arange
from os import listdir
import operator
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

# 442.38098	163.08368	26.872536	36.393852	-0.15478542	0.4350000206613913
# x1,x2 460 445 440 176

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = np.zeros((numberOfLines, 5))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    x_center = 0.0
    y_center = 0.0
    width = 0.0
    height = 0.0
    theta = 0.0
    depth = 0.0
    x1_x = 0.0
    x1_y = 0.0
    x2_x = 0.0
    x2_y = 0.0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = str(line).split('\t')
        x_center = float(listFromLine[0])
        y_center = float(listFromLine[1])
        depth = float(listFromLine[2])
    return x_center, y_center, depth


def list_to_setp(setp, list):
    for i in range(0, 9):
        setp.__dict__["input_double_register_%i" % i] = list[i]
    return setp


ROBOT_HOST = '10.149.230.20'
# ROBOT_HOST = '10.97.25.244'
ROBOT_PORT = 30004
config_filename = '/home/anbarsanti/Dropbox/YOLOv11/control/control.xml'

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe('state')
setp_names, setp_types = conf.get_recipe('setp')
watchdog_names, watchdog_types = conf.get_recipe('watchdog')

con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
connection_state = con.connect()

print("Successfully connected to the robot")

con.get_controller_version()
FREQUENCY = 500
con.send_output_setup(state_names, state_types, FREQUENCY)
setp = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)

setp.input_double_register_0 = 0
setp.input_double_register_1 = 0
setp.input_double_register_2 = 0
setp.input_double_register_3 = 0
setp.input_double_register_4 = 0
setp.input_double_register_5 = 0
setp.input_double_register_6 = 0
setp.input_double_register_7 = 0
setp.input_double_register_8 = 0
watchdog.input_double_register_0 = 0

watchdog.input_int_register_0 = 0

if not con.send_start():
    sys.exit()
last_depth = 0.5
state = con.receive()
act_q = state.actual_q

dot_q_save = []
last_dot_q = matrix([0, 0, 0, 0, 0, 0])
num = 0
f = open('/home/anbarsanti/Dropbox/YOLOv11/control/theta.txt', 'w')
f3 = open('/home/anbarsanti/Dropbox/YOLOv11/control/transferedposition.txt', 'w')
global x_cen_vec, dot_q_vec
x_cen_vec = []
y_cen_vec = []
width_vec = []
height_vec = []
dot_q_vec = []
t0 = time.time()
while (True):
    num += 1

    dt = 1
    t_now = time.time() - t0
    handle_x, handle_y, handle_z = file2matrix('/home/anbarsanti/Dropbox/YOLOv11/control/data.txt')
    x1 = int(handle_x)
    y1 = int(handle_y)
    z1 = int(handle_z)
    pot_x, pot_y, pot_z = file2matrix('/home/anbarsanti/Dropbox/YOLOv11/control/data2.txt')
    x2 = int(pot_x)
    y2 = int(pot_y)
    z2 = int(pot_z)
    spout_x, spout_y, spout_z = file2matrix('/home/anbarsanti/Dropbox/YOLOv11/control/data1.txt')
    x3 = int(spout_x)
    y3 = int(spout_y)
    z3 = int(spout_z)
    trans = matrix([[0, 0, 1, -428],
                    [-1, 0, 0, 894],
                    [0, -1, 0, 261],
                    [0, 0, 0, 1]])

    handle_cam = matrix([[x1], [y1], [z1], [1]])
    pot_cam = matrix([[x2], [y2], [z2], [1]])
    spout_cam = matrix([[x3], [y3], [z3], [1]])
    handle_rob = np.dot(trans, handle_cam)

    print('Handle Cam', handle_cam)

    print('Handle:', handle_rob)

    pot_rob = np.dot(trans, pot_cam)
    spout_rob = np.dot(trans, spout_cam)

    if x1 != 0.0:

        # print (depth)
        state = con.receive()
        pos = list([0, 0, 0, 0, 0, 0, 0, 0, 0])

        pos[0] = handle_rob[0] / 1000
        print('Pos', pos[0])
        pos[1] = (handle_rob[1] / 1000)
        print('Pos 1', pos[1])
        pos[2] = handle_rob[2] / 1000
        print('Pos 2', pos[2])
        pos[3] = pot_rob[0] / 1000
        pos[4] = pot_rob[1] / 1000
        pos[5] = pot_rob[2] / 1000
        pos[6] = spout_rob[0] / 1000
        pos[7] = spout_rob[1] / 1000
        pos[8] = spout_rob[2] / 1000
        new_setp = pos
        if pos[4] < pos[7] + 0.045:
            print('robot is watering')

        list_to_setp(setp, new_setp)
        f3.write(str(pos[0]) + '\t' + str(pos[1]) + '\t' + str(pos[2]) + '\t' + str(pos[3]) + '\t' + str(
            pos[4]) + '\t' + str(pos[5]) + str(pos[6]) + '\t' + str(pos[7]) + '\t' + str(pos[8]) + '\n')
        con.send(setp)
        print(new_setp)

        x_cen_vec = []
        y_cen_vec = []
        width_vec = []
        height_vec = []
        dot_q_vec = []

    con.send(watchdog)

con.send_pause()

con.disconnect()





