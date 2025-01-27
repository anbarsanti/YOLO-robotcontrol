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
ROBOT_HOST = "10.91.11.154"  # virtual machine in from linux host
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"
FREQUENCY = 1000 # send data in 500 Hz instead of default 125Hz

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
actual_p_plot = np.array(state.actual_TCP_pose)
actual_q = np.array(state.actual_q)
actual_q_plot = np.array(state.actual_q)

## =========================  UR5E MOVE TO INITIAL POSITION =========================
con, state, watchdog, setp = UR5e_start(con, state, watchdog, setp)

# ## ======================= TRACKING STARTS ==================================

# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or change to a specific camera index if needed
# 0 = web camera, 2 = depth camera

# Set the desired frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
	success, frame = cap.read()
	if success:
		# Run YOLO tracking on the frame
		results = model.track(frame, stream=True, show=True, persist=True,
									 tracker='bytetrack.yaml')  # Tracking with byteTrack
		
		# Process, extract, and visualize the results, source: https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results
		for r in results:
			annotated_frame = r.plot()
			
			if OBB == True:  # ==================== OBB Tracking Case ==============================
				# Data Extraction from object tracking with OBB format
				cls = r.obb.cls  # only applied in YOLO OBB model
				xyxyxyxyn = r.obb.xyxyxyxyn  # Normalized [x1, y1, x2, y2, x3, y3, x4, y4] OBBs. only applied in YOLO OBB model
				len_cls = len(cls)
				for i in range(len_cls):
					xyxyxyxyn_flatten = (np.array((xyxyxyxyn[i].tolist())).reshape(1, 8).tolist())[0]  # Flatten the xyxyxyxy
					detected_box = [*[(cls[i].tolist())], *(xyxyxyxyn_flatten)]  # Append class with its OBB
			
			else:  # ================= HBB Tracking Case ========================================
				# Data Extraction from object tracking with HBB format
				cls = r.boxes.cls  # Class labels for each HBB box. can't be applied in OBB
				xyxyn = r.boxes.xyxyn  # Normalized [x1, y1, x2, y2] horizontal boxes relative to orig_shape. can't be applied in OBB
				len_cls = len(cls)
				for i in range(len_cls):
					cls_i = cls[i].tolist()
					
					if cls_i == 0.0:  # Box's detected
						xyxyn_rev = xyxyn[i].tolist()
						
						# Shift the desired area to above the detected box
						xyxyn_rev[1] = xyxyn_rev[1] - 0.38
						xyxyn_rev[3] = xyxyn_rev[3] - 0.38
						
						# Define the desired box
						desired_box = [*[cls_i], *xyxyn_rev]  # First toy's box detected
						
						# Draw the desired box
						cv2.rectangle(annotated_frame, (int(desired_box[1] * 640), int(desired_box[2] * 480)),
										  (int(desired_box[3] * 640), int(desired_box[4] * 480)), (255, 130, 130), 2)
						cv2.putText(annotated_frame, "Desired Area",
										(int(desired_box[1] * 640), int(desired_box[2] * 480) - 10),
										cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 130, 130), 2)
					
					if cls_i == 1.0:  # Toy's detected
						reaching_box = [*[cls_i], *(xyxyn[i].tolist())]
				
				## ==================== UR5E =========================================
				# Send the q_dot to UR5e
				list_to_setp(setp, q_dot)
				con.send(setp)
				state = con.receive()
				actual_p = np.array(state.actual_TCP_pose)  # dimension (1,6)
				actual_q = np.array(state.actual_q)  # dimension (1,6)
				
				## ==================== CONTROLLER =========================================
				q_dot, epsilon, area = r2r_control(desired_box, reaching_box, actual_q, OBB=OBB)
				
				## =================== SAVE FOR PLOTTING AND ANALYSIS ===================================
				time_plot.append(time.time() - time_start)
				area_plot.append(area)
				epsilon_plot = np.append(epsilon_plot, epsilon, axis=1)
				actual_p_plot = np.vstack((actual_p_plot, actual_p))
				actual_q_plot = np.vstack((actual_q_plot, actual_q))
				q_dot_plot = np.append(q_dot_plot, q_dot, axis=1)
			
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
final_plotting (time_plot, actual_p_plot, actual_q_plot, q_dot_plot, area_plot, epsilon_plot)