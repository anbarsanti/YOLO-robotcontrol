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
ROBOT_HOST = "192.168.18.13"  # virtual machine in from linux host
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
area = 0
area_plot = [0]
time_plot = [0]
time_start = time.time()
q_dot = np.zeros((6, 1))
q_dot_plot = np.zeros((6, 1))
epsilon = np.zeros((6, 1))
epsilon_plot = np.zeros((6, 1))
actual_p = np.array(state.actual_TCP_pose)
new_actual_p = np.array([[0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])
new_actual_q = np.array([[0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])
actual_q = np.array(state.actual_q)

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
					
					
					# Capture the detected toy's box = desired box
					if cls_i == 0.0:  # First toy's box detected
						area = 0
						x_desired = [[np.array(xyxyn[i].tolist())[0]],[np.array(xyxyn[i].tolist())[1]]] # in image space
						new_actual_p = new_actual_p.reshape(6, 1)
						x_actual = [[new_actual_p[0][0]],[new_actual_p[1][0]]] # in task space
						delta_x = np.subtract(x_desired, x_actual)
						print("x_desired", x_desired)
						print("x_actual", x_actual)
						

						p_dot = - R_rc @ np.linalg.pinv(J_image_n(x_actual)) @ delta_x
						print("p_dot", p_dot)
						new_actual_q = new_actual_q.reshape(6,1)
						q_dot = np.array(10 * np.linalg.pinv(J_r(new_actual_q)) @ p_dot)
						print("q_dot", q_dot)
				
				## ==================== UR5E =========================================
				# Send the q_dot to UR5e
			
				list_to_setp(setp, q_dot)
				con.send(setp)
				state = con.receive()
				new_actual_p = np.array(state.actual_TCP_pose) # dimension (1,6)
				new_actual_q = np.array(state.actual_q) # dimension (1,6)
				
				## =================== SAVE FOR PLOTTING AND ANALYSIS ===================================
				time_plot.append(time.time() - time_start)
				area_plot.append(area)
				epsilon_plot = np.append(epsilon_plot, epsilon, axis=1)
				actual_p = np.vstack((actual_p, new_actual_p))
				actual_q = np.vstack((actual_q, new_actual_q))
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
final_plotting (time_plot, actual_p, actual_q, q_dot_plot, area_plot, epsilon_plot)