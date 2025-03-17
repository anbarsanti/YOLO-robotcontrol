import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import csv

epsilon_50 = pd.read_csv('/home/san/Dropbox/YOLOv11/results/plot_20250306 integrated 50/epsilon.csv')
area_50 = pd.read_csv('/home/san/Dropbox/YOLOv11/results/plot_20250306 integrated 50/area.csv')
actual_p_50 = pd.read_csv('/home/san/Dropbox/YOLOv11/results/plot_20250306 integrated 50/actual_p.csv')
actual_q_50 = pd.read_csv('/home/san/Dropbox/YOLOv11/results/plot_20250306 integrated 50/aqtual_q.csv')

epsilon_75 = pd.read_csv('/home/san/Dropbox/YOLOv11/results/plot_20250306 integrated 75/epsilon.csv')
area_75 = pd.read_csv('/home/san/Dropbox/YOLOv11/results/plot_20250306 integrated 75/area.csv')

epsilon_75_2 = pd.read_csv('/home/san/Dropbox/YOLOv11/results/plot_20250306 integrated 75_2/epsilon.csv')
area_75_2 = pd.read_csv('/home/san/Dropbox/YOLOv11/results/plot_20250306 integrated 75_2/area.csv')
actual_p_75_2 = pd.read_csv('/home/san/Dropbox/YOLOv11/results/plot_20250306 integrated 75_2/actual_p.csv')
actual_q_75_2 = pd.read_csv('/home/san/Dropbox/YOLOv11/results/plot_20250306 integrated 75_2/aqtual_q.csv')

time_plot = pd.read_csv('/home/san/Dropbox/YOLOv11/results/time_plot.csv')
# print(time.time())

q1 = actual_q_75_2.iloc[:,0]
q2 = actual_q_75_2.iloc[:,1]
q3 = actual_q_75_2.iloc[:,2]

plt.plot(time_plot, q1, label='1st Joint')
plt.plot(time_plot, q2, label='2nd Joint')
plt.plot(time_plot, q3, label='3rd Joint')
plt.xlabel('Time (s)')
plt.ylabel('Joint Velocity')
plt.legend()
plt.show()

# plt.figure()
# plt.grid()
# plt.axhline(y=0.5, color='green', linestyle='--', linewidth=1, label='A_d_min')
# plt.plot(time_plot, area_50/0.035, label="Proportion of the Area")
# plt.axhline(y=0.9, color='blue', linestyle='--', linewidth=1, label='A_d_max')
# plt.xlim(0,65)
# plt.ylabel('Area Proportion')
# plt.xlabel('Time [sec]')
# plt.legend()
# plt.show()

# x = actual_p_75_2.iloc[:,0]
# y = actual_p_75_2.iloc[:,1]
# z = actual_p_75_2.iloc[:,2]

# ax = plt.axes(projection='3d')
# ax.plot(x, y, z)
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# ax.set_title('3D Line Plot')
# plt.show()


