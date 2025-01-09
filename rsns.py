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
model = YOLO("model/yolo11-hbb-toy-12-01.pt")  # toys for HBB object tracking
# model = YOLO("model/yolo11-obb-11-16-watercan.pt") # watercan for OBB object tracking
# model = YOLO("model/yolo11n.pt") # object tracking with HBBqqqqqqqqq

if OBB == True:  # Initialization for OBB case
	desired_box = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	reaching_box = [0, 0, 0, 0, 0, 0, 0, 0, 0]
else:  # Initialization for HBB case
	desired_box = [0, 0, 0, 0, 0]
	reaching_box = [0, 0, 0, 0, 0]

## ========================= INITIALIZATION OF ROBOT COMMUNICATION  =========================
ROBOT_HOST = "10.149.230.168" # in robotics lab
# ROBOT_HOST = "10.91.37.45"  # virtual machine in from linux host
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"
FREQUENCY = 1000  # send data in 500 Hz instead of default 125Hz
# Setpoints to move the robot toqqqqqq
start_pose = [0.4, -0.6, 0, 0, 0, 0]
desired_value = [-0.2, -0.5, 0.2, 0.7, 0.3, -0.1] * 5

# Initialization of Plotting Variable
area_plot = [0]

## =========================  UR5E INITIALIZATION ==================================================
con, state, watchdog, setp = UR5e_init(ROBOT_HOST, ROBOT_PORT, FREQUENCY, config_filename)

# Initialization of Plotting Variable
time_plot = [0]
time_start = time.time()
q_dot = np.zeros((6, 1))
q_dot_plot = np.empty((6, 1))
epsilon = np.empty((6, 1))
epsilon_plot = np.empty((6, 1))
actual_p = np.array(state.actual_TCP_pose)
actual_q = np.array(state.actual_q)

## =========================  UR5E MOVE TO INITIAL POSITION =========================
con, state, watchdog, setp = UR5e_start(con, state, watchdog, setp)

# ## ======================= TRACKING FROM REALSENSE ==================================

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
while time.time()-time_start <5:
	frame = pipe.wait_for_frames()
	color_frame = frame.get_color_frame()
	
	# Convert images to numpy arrays
	color_image = np.asanyarray(color_frame.get_data())

	# Run YOLO tracking on the frame
	results = model.track(color_image, stream=True, show=True, persist=True, tracker='bytetrack.yaml')  # Tracking with byteTrack

# Start Tracking while communication with the robot
while True:
	frame = pipe.wait_for_frames()
	color_frame = frame.get_color_frame()
	
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

				# Capture the first detected toy's box = desired box
				if cls_i == 0.0 and desired_box == [0, 0, 0, 0, 0]:
					desired_box = [*[cls_i], *(xyxyn[i].tolist())] # First toy's box detected
						
				# Detect the reaching box = the toy
				if cls_i == 1.0:
					reaching_box = [*[cls_i], *(xyxyn[i].tolist())]

			# Draw the desired box if it already exists
			if desired_box != [0, 0, 0, 0, 0]:
				cv2.rectangle(annotated_frame, (int(desired_box[1] * 640), int(desired_box[2] * 480)), (int(desired_box[3] * 640), int(desired_box[4] * 480)), (0,255,0), 2)
				cv2.putText(annotated_frame, "Desired Box", (int(desired_box[1] * 640), int(desired_box[2] * 480)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

			area = intersection_area_HBB_xyxy(desired_box, reaching_box)
				
			## ==================== UR5E =========================================
			# Send the q_dot to UR5e
			list_to_setp(setp, q_dot)
			con.send(setp)
			state = con.receive()
			new_actual_p = np.array(state.actual_TCP_pose)
			new_actual_q = np.array(state.actual_q)
			
			## ==================== CONTROLLER =========================================
			q_dot, epsilon = r2r_control(desired_box, reaching_box, new_actual_q, OBB=OBB)
				
			## =================== SAVE FOR PLOTTING AND ANALYSIS ===================================
			time_plot.append(time.time()-time_start)
			area_plot.append(area)
			epsilon_plot = np.append(epsilon_plot, epsilon, axis=1)
			actual_p = np.vstack((actual_p, new_actual_p))
			actual_q = np.vstack((actual_q, new_actual_q))
			q_dot_plot = np.append(q_dot_plot, q_dot, axis=1)
				
		# Display the annotated frame
		cv2.imshow("YOLOv11 Tracking - Realsense", annotated_frame)
		
	# Break the loop if 'q' is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Stop Streaming
pipe.stop()
cv2.destroyAllWindows()

## =========================  DISCONNECTING THE UR5E ========================================
con.send(watchdog)
con.send_pause()
con.disconnect()

## =========================  FINAL PLOTTING ==================================================
final_plotting(time_plot, actual_p, actual_q, q_dot_plot, area_plot, epsilon_plot)
