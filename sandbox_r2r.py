"""
﷽
by @anbarsanti
"""
import sys
sys.path.append('../RTDE_Python_Client_Library')
from r2r_functions import *
import numpy as np

## ====================== INITIALIZATION OF TRACKING STUFF ==================================
OBB = True
# model = YOLO("model/yolo11-hbb-toy-25-02-24.pt") # toys for HBB object tracking
model = YOLO("model/yolo11-obb-25-03-04-watercan_best.pt") # watercan for OBB object tracking
# model = YOLO("model/yolo11n.pt") # object tracking with HBB

if OBB==True: # Initialization for OBB case
	desired_box = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	reaching_box = [0, 0, 0, 0, 0, 0, 0, 0, 0]
else: # Initialization for HBB case
	desired_box = [0, 0, 0, 0, 0]
	reaching_box = [0, 0, 0, 0, 0]

## ========================= INITIALIZATION OF ROBOT COMMUNICATION  =========================
ROBOT_HOST = "10.149.230.1" # in robotics lab
# ROBOT_HOST = "192.168.18.13"  # virtual machine in from linux host
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"
FREQUENCY = 1000 # send data in 500 Hz instead of default 125Hz

## =========================  UR5E INITIALIZATION ==================================================
con, state, watchdog, setp = UR5e_init(ROBOT_HOST, ROBOT_PORT, FREQUENCY, config_filename)

# Initialization of Plotting Variable
area_proportion = 0
area_proportion_plot = [0]
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
x_actual = [[0],[0]]
x_desired = [[0],[0]]

## =========================  UR5E MOVE TO INITIAL POSITION =========================
con, state, watchdog, setp = UR5e_start(con, state, watchdog, setp)

## ======================= TRACKING STARTS FROM INTEL REALSENSE==================================

# Check RealSense Camera Connection
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
	print("No device connected")
else:
	print("device connected")
for dev in devices:
	print(dev.get_info(rs.camera_info.name))

# Initialize RealSense Pipeline
pipe = rs.pipeline()
cfg = rs.config()

# Enable color stream (and depth if you want)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

# Start Streaming
pipe.start(cfg)

# Wait until the realsense stable
while time.time() - time_start < 8:
	frame = pipe.wait_for_frames()
	color_frame = frame.get_color_frame()
	# Convert images to numpy arraysq
	color_image = np.asanyarray(color_frame.get_data())

	# Run YOLO tracking on the frame
	results = model.track(color_image, stream=True, show=True, persist=True,
								 tracker='bytetrack.yaml')  # Tracking with byteTrack

# Start Tracking while communication with the robot
while True:
	frame = pipe.wait_for_frames()
	color_frame = frame.get_color_frame()
	depth_frame = frame.get_depth_frame()

	# Convert images to numpy arrays
	color_image = np.asanyarray(color_frame.get_data())

	# Run YOLO tracking on the frame
	results = model.track(color_image, stream=True, show=True, persist=True,
								 tracker='bytetrack.yaml')  # Tracking with byteTrack

	# Process, extract, and visualize the results, source: https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results
	for r in results:
		annotated_frame = r.plot()

		if OBB == True:  # ==================== OBB Tracking Case ==============================
			# Data Extraction from object tracking with OBB format
			cls = r.obb.cls  # class labels for each OBB box, only applied in YOLO OBB model
			xyxyxyxyn = r.obb.xyxyxyxyn  # Normalized [x1, y1, x2, y2, x3, y3, x4, y4] OBBs. only applied in YOLO OBB model
			len_cls = len(cls)
			for i in range(len_cls):
				cls_i = cls[i].tolist()
				cls_name = model.names[cls_i]

				if cls_name == "pot":
					xyxyxyxyn_d = (np.array((xyxyxyxyn[i].tolist())).reshape(1, 8).tolist())[0]  # Flatten the xyxyxyxy

					# Shift the desired region to above the detected box
					xyxyxyxyn_d[0] = xyxyxyxyn_d[0] + 0.20
					xyxyxyxyn_d[1] = xyxyxyxyn_d[1] - 0.30
					xyxyxyxyn_d[2] = xyxyxyxyn_d[2] + 0.20
					xyxyxyxyn_d[3] = xyxyxyxyn_d[3] - 0.30
					xyxyxyxyn_d[4] = xyxyxyxyn_d[4] + 0.20
					xyxyxyxyn_d[5] = xyxyxyxyn_d[5] - 0.30
					xyxyxyxyn_d[6] = xyxyxyxyn_d[6] + 0.20
					xyxyxyxyn_d[7] = xyxyxyxyn_d[7] - 0.30

					# Define the desired box
					desired_box = [*[cls_i], *xyxyxyxyn_d]  # Append class with its OBB

					# Calculate the area of desired region
					desired_vertices = convert_OBB_to_vertices(desired_box)
					area_desired = Polygon(desired_vertices).area
					print("area_desired = ", area_desired)

					# Desired box's depth
					x_d = int((xyxyxyxyn_d[0]+xyxyxyxyn_d[2])*320)
					y_d = int((xyxyxyxyn_d[1]+xyxyxyxyn_d[3])*240)
					if depth_frame:
						desired_depth = depth_frame.get_distance(x_d, y_d)
					else:
						desired_depth = 0.75

					# Draw the desired box
					cv2.rectangle(annotated_frame, (int(desired_box[1] * 640), int(desired_box[2] * 480)),
									  (int(desired_box[5] * 640), int(desired_box[6] * 480)), (255, 220, 220), 2)
					cv2.putText(annotated_frame, "Desired Area",
									(int(desired_box[1] * 640), int(desired_box[2] * 480) - 10),
									cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 220), 2)

				if cls_name == "water can":  # water can is detected
					xyxyxyxyn_r = (np.array((xyxyxyxyn[i].tolist())).reshape(1, 8).tolist())[0]  # Flatten the xyxyxyxy

					# Define the reaching box
					reaching_box = [*[cls_i], *xyxyxyxyn_r]
					print("reaching_box", reaching_box)

					# Calculate the area of reaching box
					reaching_vertices = convert_OBB_to_vertices(reaching_box)
					area_reaching = Polygon(reaching_vertices).area
					print("area_reaching = ", area_reaching)

					# Reaching box's depth'
					x_r = int((xyxyxyxyn_r[0] + xyxyxyxyn_r[2]) * 320)
					y_r = int((xyxyxyxyn_r[1] + xyxyxyxyn_r[3]) * 240)
					# if depth_frame:
					# 	reaching_depth = depth_frame.get_distance(x_r, y_r)
					# else:
					# 	reaching_depth = 0.75

		else:  # ================= HBB Tracking Case ========================================
			# Data Extraction from object tracking with HBB format
			cls = r.boxes.cls  # Class labels for each HBB box. can't be applied in OBB
			xyxyn = r.boxes.xyxyn  # Normalized [x1, y1, x2, y2] horizontal boxes relative to orig_shape. can't be applied in OBB
			len_cls = len(cls)
			for i in range(len_cls):
				cls_i = cls[i].tolist()

				if cls_i == 0.0: # Box's detected
					xyxyn_d = xyxyn[i].tolist()

					# Shift the desired area to above the detected box
					# xyxyn_d[0] = xyxyn_d[0] - 0.1 # For Scaling
					xyxyn_d[1] = xyxyn_d[1] - 0.20 # For Scaling
					# xyxyn_d[2] = xyxyn_d[2] + 0.1 # For Scaling
					xyxyn_d[3] = xyxyn_d[3] - 0.20 # For Scaling

					# Define the desired box
					desired_box = [*[cls_i], *xyxyn_d]  # First toy's box detected

					# Desired box's depth
					x_d = int((xyxyn_d[0]+xyxyn_d[2])*320)
					y_d = int((xyxyn_d[1]+xyxyn_d[3])*240)
					if depth_frame:
						desired_depth = depth_frame.get_distance(x_d, y_d)
					else:
						desired_depth = 0.75

					# Draw the desired box
					cv2.rectangle(annotated_frame, (int(desired_box[1] * 640), int(desired_box[2] * 480)),
									  (int(desired_box[3] * 640), int(desired_box[4] * 480)), (255, 255, 255), 2)
					cv2.putText(annotated_frame, "Desired Area",
									(int(desired_box[1] * 640), int(desired_box[2] * 480) - 10),
									cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

				if cls_i == 1.0: # Toy's detected
					xyxyn_r = xyxyn[i].tolist()
					reaching_box = [*[cls_i], *(xyxyn[i].tolist())]

					# Reaching box's depth'
					x_r = int((xyxyn_r[0]+xyxyn_r[2])*320)
					y_r = int((xyxyn_r[1]+xyxyn_r[3])*240)
					if depth_frame:
						reaching_depth = depth_frame.get_distance(x_r, y_r)
					else:
						reaching_depth = 0.75

			## ==================== UR5E =========================================
			# Send the q_dot to UR5e
			list_to_setp(setp, q_dot)
			con.send(setp)
			state = con.receive()
			actual_p = np.array(state.actual_TCP_pose) # dimension (1,6)
			actual_q = np.array(state.actual_q) # dimension (1,6)qqqq

			## ==================== CONTROLLER =========================================
			q_dot, epsilon, area_proportion = r2r_control(reaching_box, desired_box, reaching_depth, desired_depth, actual_q, OBB=OBB)

			## =================== SAVE FOR PLOTTING AND ANALYSIS ===================================
			time_plot.append(time.time() - time_start)
			area_proportion_plot.append(area_proportion)
			epsilon_plot = np.append(epsilon_plot, epsilon, axis=1)
			actual_p_plot = np.vstack((actual_p_plot, actual_p))
			actual_q_plot = np.vstack((actual_q_plot, actual_q))
			q_dot_plot = np.append(q_dot_plot, q_dot, axis=1)

		# Display the annotated frame
		cv2.imshow("YOLOv11 Tracking - Realsense", annotated_frame)

	# Break the loop if 'q' is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# # Release resources (Webcam)
# cap.release()
# cv2.destroyAllWindows()

# Stop Streaming (IntelRealsense)
pipe.stop()
cv2.destroyAllWindows()

## =========================  DISCONNECTING THE UR5E ========================================
con.send(watchdog)
con.send_pause()
con.disconnect()

## =========================  FINAL PLOTTING ==================================================
final_plotting (time_plot, actual_p_plot, actual_q_plot, q_dot_plot, area_plot, area_reaching_plot, area_desired_plot, epsilon_plot)