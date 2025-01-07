"""
﷽
by @anbarsanti
"""
import sys
sys.path.append('../RTDE_Python_Client_Library')
from r2r_functions import *
import numpy as np

## ====================================  HBB TEST ====================================
## print("==================================== HBB Test ==================================== ")
# hbbA_xywh = np.array([1, 0.7, 0.3, 0.4, 0.4]) #xywh format
# hbbB_xywh = np.array([1, 0.8, 0.2, 0.2, 0.2]) #xywh format
# hbbC_xywh = np.array([1, 0.35, 0.6, 0.5, 0.4]) #xywh format
# hbbD_xyxy = np.array([0, 0.2, 0.2, 0.8, 0.8])
# hbbE_xyxy = np.array([0, 0.3, 0.1, 0.5, 0.9])
# hbbF_xyxy = np.array([0, 0.6, 0.4, 0.9, 0.6])
#
# print("area HBB A and B", intersection_area_HBB_xywh(hbbA_xywh, hbbB_xywh))
# print("area HBB A and C", intersection_area_HBB_xywh(hbbA_xywh, hbbC_xywh))
# print("area HBB B and C", intersection_area_HBB_xywh(hbbB_xywh, hbbC_xywh))
#
# print("area HBB D and E", intersection_area_HBB_xyxy(hbbD_xyxy, hbbE_xyxy))
# print("area HBB D and F", intersection_area_HBB_xyxy(hbbD_xyxy, hbbF_xyxy))
# print("area HBB E and F", intersection_area_HBB_xyxy(hbbE_xyxy, hbbF_xyxy))

# print("check vertices hbbA_xywh", convert_HBB_xywh_to_vertices(hbbA_xywh))

# reaching_box = [0.0, 0.59080827236, 0.2633193731079834, 0.844, 0.617902934551239] # Toys
# desired_box = [1.0, 0.36258264, 0.5401297211647, 0.5640562176704407, 0.8815667033195496] # Box
# hbbF = [1, 0,0,0,0]
# print(is_HBB_intersect(reaching_box, desired_box))
# print("Area between reaching box and desired box", intersection_area_HBB(reaching_box, desired_box))

# print("Area A vs D", intersection_area_HBB(hbbA, hbbC))
# print("Area B vs D", intersection_area_HBB(hbbB, hbbC))
# print("image feature points of hbbA:", cxywh2xyxyxy(hbbA))

### ====================================  OBB ====================================
# print("==================================== OBB Test ==================================== ")
# obbA = np.array([0, 0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8])
# obbB = np.array([0, 0.5, 0.9, 0.9, 0.9, 0.9, 1.0, 0.5, 1.0])
# obbC = np.array([0, 0.2, 0.1, 0.4, 0.1, 0.4, 0.9, 0.2, 0.9])
# obbD = np.array([0, 0.6, 0.1, 0.9, 0.4, 0.7, 0.8, 0.4, 0.5])
# obbE = np.array([0, 0.2, 0.1, 0.4, 0.3, 0.3, 0.4, 0.1, 0.2])
# print("Area of A and C", intersection_area_OBB_diy(obbA, obbC))
# print("Area of A and C shapely", intersection_area_OBB_shapely(obbA, obbC))
# print("Area of A and D", intersection_area_OBB_diy(obbA, obbD))
# print("Area of A and D shapely", intersection_area_OBB_shapely(obbA, obbD))
# print("Area of A and E", intersection_area_OBB_diy(obbA, obbE))
# print("Area of A and E shapely", intersection_area_OBB_shapely(obbA, obbE))
# print("Area of C and D", intersection_area_OBB_diy(obbC, obbD))
# print("Area of C and D shapely", intersection_area_OBB_shapely(obbC, obbD))
# print("Area of C and E", intersection_area_OBB_diy(obbC, obbE))
# print("Area of C and E shapely", intersection_area_OBB_shapely(obbC, obbE))
# p1 = intersection_points_OBB_diy(obbA, obbC)
# print("interp A and C:", p1)
# p2 = intersection_points_OBB_diy(obbA, obbD)
# print("interp A and D:", p2)
# p3 = intersection_points_OBB_diy(obbA, obbE)
# print("interp A and E:", p3)
# p4 = intersection_points_OBB_diy(obbC, obbD)
# print("interp C and D:", p4)
# p5 = intersection_points_OBB_diy(obbC, obbE)
# print("interp C and E:", p5)

# # # ------------------------------ Jacobian Test ---------------------------------
# # print("----------------------Jacobian Test--------------------------")
# q = np.array([0.23, 0.91, 0.22, 0.12, 0.42, 0.74]).reshape((-1,1))
# q_dot = r2r_control(obbD, obbE, q, OBB=True)

## ======================================= INTEGRATION TEST ============================================
## ====================== INITIALIZATION OF TRACKING STUFF ==================================
OBB = False
model = YOLO("model/yolo11-hbb-toy-12-01.pt") # toys for HBB object tracking
# model = YOLO("model/yolo11-obb-11-16-watercan.pt") # watercan for OBB object tracking
# model = YOLO("model/yolo11n.pt") # object tracking with HBB

if OBB==True: # Initialization for OBB case
	desired_box = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	reaching_box = [0, 0, 0, 0, 0, 0, 0, 0, 0]
else: # Initialization for HBB case
	desired_box = [0, 0, 0, 0, 0]
	reaching_box = [0, 0, 0, 0, 0]
	
r_desired_box = None

## ========================= INITIALIZATION OF ROBOT COMMUNICATION  =========================
# ROBOT_HOST = "10.149.230.168" # in robotics lab
ROBOT_HOST = "192.168.18.14"  # virtual machine in from linux host
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"
FREQUENCY = 500 # send data in 500 Hz instead of default 125Hz
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
# actual_qd = np.array(state.actual_qd)
# con, state, watchdog, setp, actual_p, actual_q, actual_qd= UR5e_loop(con, state, watchdog, setp, desired_value,
# 																							time_start, trajectory_time, time_plot,
# 																							actual_p, actual_q, actual_qd)

# ## ======================= TRACKING STARTS ==================================
# Initialize locked_box, the box that is locked
# in HBB toy detection, locked_box is the red box with class 1

# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or change to a specific camera index if needed
# 0 = web camera, 2 = depth camera

# Set the desired frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
	success, frame = cap.read()
	if success:
		# Run YOLOv8 OBB tracking on the frame. Tracking also can be used for OBB
		# persist = True --> to maintain track continuity between frames
		results = model.track(frame, stream=True, show=True, persist=True,
									 tracker='bytetrack.yaml')  # Tracking with byteTrack

		# Process, extract, and visualize the results
		# Source: https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results
		
		for r in results:
			annotated_frame = r.plot()
			
			# Plot the annotated desired box
			if r_desired_box is not None:
				print("r_desired_box", r_desired_box)
				annotated_desired_box = r_desired_box.plot()
			
			if OBB == True:
				# Data Extraction from object tracking with OBB format
				cls = r.obb.cls  # only applied in YOLO OBB model
				xyxyxyxyn = r.obb.xyxyxyxyn  # Normalized [x1, y1, x2, y2, x3, y3, x4, y4] OBBs. only applied in YOLO OBB model
				len_cls = len(cls)
				for i in range(len_cls):
					xyxyxyxyn_flatten = (np.array((xyxyxyxyn[i].tolist())).reshape(1, 8).tolist())[0]  # Flatten the xyxyxyxy
					detected_box = [*[(cls[i].tolist())], *(xyxyxyxyn_flatten)]  # Append class with its OBB
			
			else:  # HBB
				# Data Extraction from object tracking with HBB format
				cls = r.boxes.cls  # Class labels for each HBB box. can't be applied in OBB
				xyxyn = r.boxes.xyxyn  # Normalized [x1, y1, x2, y2] horizontal boxes relative to orig_shape. can't be applied in OBB
				len_cls = len(cls)
				for i in range(len_cls):
					cls_i = cls[i].tolist()
					detected_box = [*[cls_i], *(xyxyn[i].tolist())]  # Append class with its HBB
					print("detected box", detected_box)
					
					# Capture the first detected toy's box
					if cls_i == 0.0 and desired_box == [0, 0, 0, 0, 0]:
						print("first box detected different than zeros")
						r_desired_box = r
						desired_box = detected_box
						
				# # Send the desired value to UR5e
				# list_to_setp(setp, desired_value)
				# con.send(setp)
				# state = con.receive()
				# new_actual_p = np.array(state.actual_TCP_pose)
				# new_actual_q = np.array(state.actual_q)
				# # new_actual_qd = np.array(state.actual_qd)
				# #
				# # # Plotting Purpose
				# time_plot.append(time.time() - time_start)
				# actual_p = np.vstack((actual_p, new_actual_p))
				# actual_q = np.vstack((actual_q, new_actual_q))
				# # actual_qd = np.vstack((actual_qd, new_actual_qd))
			
			# Display the annotated frame
			cv2.imshow("YOLOv11 Tracking - Webcam", annotated_frame)
			cv2.imshow(annotated_desired_box)
		
		# Break the loop if 'q' is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

# Release resources
cap.release()
cv2.destroyAllWindows()

## =========================  DISCONNECTING THE UR5E ========================================
con.send(watchdog)
con.send_pause()
con.disconnect()

## =========================  FINAL PLOTTING ==================================================
final_plotting(time_plot, actual_p, actual_q)#, actual_qd)
