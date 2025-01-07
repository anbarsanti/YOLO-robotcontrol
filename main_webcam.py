"""
﷽
by @anbarsanti
"""
import sys
sys.path.append('../RTDE_Python_Client_Library')
from r2r_functions import *
import numpy as np

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
time_plot = [0]
actual_p = np.array(state.actual_TCP_pose)
actual_q = np.array(state.actual_q)
q_dot = np.zeros((6,1))

## =========================  UR5E MOVE TO INITIAL POSITION =========================
con, state, watchdog, setp = UR5e_start(con, state, watchdog, setp)

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
			
			if OBB == True: # OBB Tracking
				# Data Extraction from object tracking with OBB format
				cls = r.obb.cls  # only applied in YOLO OBB model
				xyxyxyxyn = r.obb.xyxyxyxyn  # Normalized [x1, y1, x2, y2, x3, y3, x4, y4] OBBs. only applied in YOLO OBB model
				len_cls = len(cls)
				for i in range(len_cls):
					xyxyxyxyn_flatten = (np.array((xyxyxyxyn[i].tolist())).reshape(1, 8).tolist())[0]  # Flatten the xyxyxyxy
					detected_box = [*[(cls[i].tolist())], *(xyxyxyxyn_flatten)]  # Append class with its OBB
			
			else:  # HBB Tracking
				# Data Extraction from object tracking with HBB format
				cls = r.boxes.cls  # Class labels for each HBB box. can't be applied in OBB
				xyxyn = r.boxes.xyxyn  # Normalized [x1, y1, x2, y2] horizontal boxes relative to orig_shape. can't be applied in OBB
				len_cls = len(cls)
				for i in range(len_cls):
					cls_i = cls[i].tolist()
					
					# Capture the first detected toy's box = desired box
					if cls_i == 0.0 and desired_box == [0, 0, 0, 0, 0]:
						desired_box = [*[cls_i], *(xyxyn[i].tolist())] # First toy's box detected
						# print("desired box", desired_box)

					# Detect the reaching box = the toy
					if cls_i == 1.0:
						reaching_box = [*[cls_i], *(xyxyn[i].tolist())]
						# print("reaching box", reaching_box)
				
				# Draw the desired box if it already exists
				if desired_box != [0, 0, 0, 0, 0]:
					cv2.rectangle(annotated_frame, (int(desired_box[1] * 640), int(desired_box[2] * 480)), (int(desired_box[3] * 640), int(desired_box[4] * 480)), (0,255,0), 2)
					cv2.putText(annotated_frame, "Desired Box", (int(desired_box[1] * 640), int(desired_box[2] * 480)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
				
				area = intersection_area_HBB_xyxy(desired_box, reaching_box)
				
				# Send the q_dot to UR5e
				list_to_setp(setp, q_dot)
				con.send(setp)
				state = con.receive()
				new_actual_p = np.array(state.actual_TCP_pose)
				new_actual_q = np.array(state.actual_q)
				
				# Controller
				q_dot = r2r_control(desired_box, reaching_box, new_actual_q, OBB=OBB)
				print("q_dot", q_dot)
				
				# # # Plotting Purpose
				# time_plot.append(time.time() - time_start)
				# actual_p = np.vstack((actual_p, new_actual_p))
				# actual_q = np.vstack((actual_q, new_actual_q))

			
			# Display the annotated frame
			cv2.imshow("YOLOv11 Tracking - Webcam", annotated_frame)
			
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
final_plotting(time_plot, actual_p, actual_q)
