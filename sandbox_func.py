"""
﷽
by @anbarsanti
"""
import sys
sys.path.append('../RTDE_Python_Client_Library')
from r2r_functions import *
import numpy as np

## ======================================================== REALSENSE TESTING =============================================================
# OBB = False
# model = YOLO("model/yolo11-hbb-toy-12-01.pt")  # toys for HBB object tracking
#
# # Check RealSense Camera Connection
# ctx = rs.context()
# devices = ctx.query_devices()
#
# if len(devices) == 0:
# 	print("No device connected")
# else:
# 	print("device connected")
# for dev in devices:
# 	print(dev.get_info(rs.camera_info.name))
#
# # Initialize RealSense Pipeline
# pipe = rs.pipeline()
# cfg = rs.config()
#
# # Enable color stream (and depth if you want)
# cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# # cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
#
# # Start Streaming
# pipe.start(cfg)
#
# while True:
# 	frame = pipe.wait_for_frames()
# 	color_frame = frame.get_color_frame()
#
# 	# Convert images to numpy arrays
# 	color_image = np.asanyarray(color_frame.get_data())
#
# 	# # Show Images
# 	# cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
# 	# cv2.imshow("Bismillah", color_image)
#
# 	# Run YOLO tracking on the frame
# 	results = model.track(color_image, stream=True, show=True, persist=True,
# 								 tracker='bytetrack.yaml')  # Tracking with byteTrack
#
# 	# Process and visualize the results of Object Tracking with YOLO (BUGS IN THIS PART)
# 	for r in results:
# 		annotated_frame = r.plot()
#
# 		if OBB == True:
# 			# Data Extraction from object tracking with OBB format
# 			cls = r.obb.cls  # class labels for each OBB box, only applied in YOLO OBB model
# 			xyxyxyxyn = r.obb.xyxyxyxyn  # Normalized [x1, y1, x2, y2, x3, y3, x4, y4] OBBs. only applied in YOLO OBB model
# 			len_cls = len(cls)
# 			for i in range(len_cls):
# 				xyxyxyxyn_flatten = (np.array((xyxyxyxyn[i].tolist())).reshape(1, 8).tolist())[0]  # Flatten the xyxyxyxy
# 				detected_box = [*[(cls[i].tolist())], *(xyxyxyxyn_flatten)]  # Append class with its OBB
#
# 		else:  # HBB
# 			# Data Extraction from object tracking with HBB format
# 			cls = r.boxes.cls  # Class labels for each HBB box. can't be applied in OBB
# 			xyxyn = r.boxes.xyxyn  # Normalized [x1, y1, x2, y2] horizontal boxes relative to orig_shape. can't be applied in OBB
# 			len_cls = len(cls)
# 			for i in range(len_cls):
# 				detected_box = [*[(cls[i].tolist())], *(xyxyn[i].tolist())]  # Append class with its HBB
#
# 		# Display the annotated frame
# 		cv2.imshow("YOLOv11 OBB Inference - Realsense Camera", annotated_frame)
#
# 	# Break the loop if 'q' is pressed
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break
#
# # Stop Streaming
# pipe.stop()
# cv2.destroyAllWindows()

## ======================================================== WEBCAM TESTING ==============================================================
# # Open the camera
# cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or change to a specific camera index if needed
# # 0 = web camera, 2 = depth camera
#
# # Set the desired frame width and height
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
# while cap.isOpened():
# 	success, frame = cap.read()
# 	if success:
# 		# Run YOLOv8 OBB tracking on the frame. Tracking also can be used for OBB
# 		# persist = True --> to maintain track continuity between frames
# 		results = model.track(frame, stream=True, show=True, persist=True,
# 									 tracker='bytetrack.yaml')  # Tracking with byteTrack
#
# 		# Process, extract, and visualize the results
# 		for r in results:
# 			annotated_frame = r.plot()
#
# 			# Results Documentation:
# 			# https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results
#
# 			if OBB == True:
# 				# Data Extraction from object tracking with OBB format
# 				cls = r.obb.cls  # only applied in YOLO OBB model
# 				xyxyxyxyn = r.obb.xyxyxyxyn  # Normalized [x1, y1, x2, y2, x3, y3, x4, y4] OBBs. only applied in YOLO OBB model
# 				len_cls = len(cls)
# 				for i in range(len_cls):
# 					xyxyxyxyn_flatten = (np.array((xyxyxyxyn[i].tolist())).reshape(1, 8).tolist())[0]  # Flatten the xyxyxyxy
# 					detected_box = [*[(cls[i].tolist())], *(xyxyxyxyn_flatten)]  # Append class with its OBB
#
# 			else:  # HBB
# 				# Data Extraction from object tracking with HBB format
# 				cls = r.boxes.cls  # Class labels for each HBB box. can't be applied in OBB
# 				xyxyn = r.boxes.xyxyn  # Normalized [x1, y1, x2, y2] horizontal boxes relative to orig_shape. can't be applied in OBB
# 				len_cls = len(cls)
# 				for i in range(len_cls):
# 					detected_box = [*[(cls[i].tolist())], *(xyxyn[i].tolist())]  # Append class with its HBB
#
# 			# Display the annotated frame
# 			cv2.imshow("YOLOv11 Tracking - Webcam", annotated_frame)
#
# 		# Break the loop if 'q' is pressed
# 		if cv2.waitKey(1) & 0xFF == ord('q'):
# 			break
# 	else:
# 		break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()

## ===================================================  HBB TESTING ===================================================
# hbbA_xywh = np.array([1, 0.7, 0.3, 0.4, 0.4]) #xywh format
# hbbB_xywh = np.array([1, 0.8, 0.2, 0.2, 0.2]) #xywh format
# hbbC_xywh = np.array([1, 0.35, 0.6, 0.5, 0.4]) #xywh format
hbbD_xyxy = np.array([0, 0.2, 0.2, 0.8, 0.8])
hbbE_xyxy = np.array([0, 0.3, 0.1, 0.5, 0.9])
hbbF_xyxy = np.array([0, 0.6, 0.4, 0.9, 0.6])
# #
# # print("area HBB A and B", intersection_area_HBB_xywh(hbbA_xywh, hbbB_xywh))
# # print("area HBB A and C", intersection_area_HBB_xywh(hbbA_xywh, hbbC_xywh))
# # print("area HBB B and C", intersection_area_HBB_xywh(hbbB_xywh, hbbC_xywh))
# #
# # print("area HBB D and E", intersection_area_HBB_xyxy(hbbD_xyxy, hbbE_xyxy))
# # print("area HBB D and F", intersection_area_HBB_xyxy(hbbD_xyxy, hbbF_xyxy))
# # print("area HBB E and F", intersection_area_HBB_xyxy(hbbE_xyxy, hbbF_xyxy))
#
# print("hbbD in xywhr format", cxyxy2xywhr(hbbD_xyxy))
# print("hbbE in xywhr format", cxyxy2xywhr(hbbE_xyxy))
# print("hbbF in xywhr format", cxyxy2xywhr(hbbF_xyxy))
# #
# q = [    0.39781,    -0.60547,   0.0021877,   0.0076568,   0.0032815,  -0.0010938]
# # q_reshape = np.array(q).reshape((-1,1))
# # print("q_reshape", q_reshape)
# transposed = np.array(q).reshape(-1,1)
# print(transposed)
#
#
#
#
# ### ====================================  OBB TESTING ===================================================
# obbA = np.array([0, 0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8])
# obbB = np.array([0, 0.5, 0.9, 0.9, 0.9, 0.9, 1.0, 0.5, 1.0])
# obbC = np.array([0, 0.2, 0.1, 0.4, 0.1, 0.4, 0.9, 0.2, 0.9])
# obbD = np.array([0, 0.6, 0.1, 0.9, 0.4, 0.7, 0.8, 0.4, 0.5])
# obbE = np.array([0, 0.2, 0.1, 0.4, 0.3, 0.3, 0.4, 0.1, 0.2])
# # print("Area of A and C", intersection_area_OBB_diy(obbA, obbC))
# print("Area of A and C shapely", intersection_area_OBB_shapely(obbA, obbC))
# print("Area of A and D", intersection_area_OBB_diy(obbA, obbD))
# print("Area of A and D shapely", intersection_area_OBB_shapely(obbA, obbD))
# print("Area of A and E", intersection_area_OBB_diy(obbA, obbE))
# print("Area of A and E shapely", intersection_area_OBB_shapely(obbA, obbE))
# print("Area of C and D", intersection_area_OBB_diy(obbC, obbD))
# print("Area of C and D shapely", intersection_area_OBB_shapely(obbC, obbD))
# print("Area of C and E", intersection_area_OBB_diy(obbC, obbE))
# print("Area of C and E shapely", intersection_area_OBB_shapely(obbC, obbE))
# sorted_p1 = sort_points_clockwise(p1)
# print("sorted_p1", sorted_p1)
# p2 = intersection_points_OBB_diy(obbA, obbD)
# print("interp A and D:", p2)
# p3 = intersection_points_OBB_diy(obbA, obbE)
# print("interp A and E:", p3)
# p4 = intersection_points_OBB_diy(obbC, obbD)
# print("interp C and D:", p4)
# p5 = intersection_points_OBB_diy(obbC, obbE)
# print("interp C and E:", p5)

# # # # ================================= JACOBIAN TESTING ================================================
# q = np.array([0.23, 0.91, 0.22, 0.12, 0.42, 0.74]).reshape((-1,1))
# p_r_hbb = cxyxy2xyxyxy(hbbD_xyxy)
# p_r_obb = cxyxyxyxy2xyxyxy(obbB)
#
# # print("q_dot", q_dot)
# # print("epsilon_A", epsilon_A)
# # print("epsilon_S", epsilon_S)
# # print("J_o_I_r", J_o_I_r)
# # print("J_alpha_a_r", J_alpha_a_r)
#
# # ## ===================== J_o_I_r Testing =====================
# # print("J_o(p_r_box)", J_o(p_r_obb))
# # print("J_o(p_r_box).shape", J_o(p_r_obb).shape)
# # print("J_I(p_r_box)", J_I(p_r_obb))
# # print("J_I(p_r_box).shape", J_I(p_r_obb).shape)
# # print("J_r(q)", J_r(q))
# # print("J_r(q).shape", J_r(q).shape)
# J_o_I_r = (J_o(p_r_obb)) @ (J_I(p_r_obb)) @ (J_r(q))
# # print("J_o_I_r", J_o_I_r)
# print("J_o_I_r.shape", J_o_I_r.shape)
# J_o_I_r_pinv = np.linalg.pinv(J_o_I_r)
# print("J_o_I_r_pinv", J_o_I_r_pinv)
# print("J_o_I_r_pinv.shape", J_o_I_r_pinv.shape)
# # J_o_I_r_transpose = J_o_I_r.T
# print("J_o_I_r_transpose", J_o_I_r_transpose)
# print("J_o_I_r_transpose.shape", J_o_I_r_transpose.shape)

# ## ===================== J_olpha_a_r Testing =====================
# p1 = intersection_points_HBB_xyxy(hbbD_xyxy, hbbF_xyxy)
# # # print("interp D and E:", p1)
# # # # print("J_alpha(p1)", J_alpha(p1))
# # # # print("J_alpha(p1).shape", J_alpha(p1).shape)
# print("J_a_n(p1)", J_a(p1))
# print("J_a_n(p1).shape", J_a(p1).shape)
# # # print("J_r(q)", J_r(q))
# # # print("J_r(q).shape", J_r(q).shape)
# J_alpha_a_r = ((J_alpha(p1)) @ (J_a(p1)) @ (J_r(q))).reshape(1,6)
# print("J_alpha_a_r.shape", J_alpha_a_r.shape)
# J_alpha_a_r_pinv = np.linalg.pinv(J_alpha_a_r)
# print("J_alpha_a_r_pinv", J_alpha_a_r_pinv)
# print("J_alpha_a_r_pinv.shape", J_alpha_a_r_pinv.shape)
#
# jacobian = np.concatenate((J_alpha_a_r_pinv, J_o_I_r_pinv), axis=1)
# print("jacobian", jacobian)
# print("jacobian.shape", jacobian.shape)
#
#
# q_dot, epsilon= r2r_control(obbD, obbE, q, OBB=True)
# print("epsilon", epsilon)
# print("epsilon.shape", epsilon.shape)
# print("q_dot", q_dot)
# print("q_dot.shape", q_dot.shape)
#
## ========================================= IMAGE JACOBIAN TESTING ==============================================================
x0 = [[0.5],[0.3]]
x1 = [[0.5],[0.9]]
x2 = [[0.8],[0.7]]
x3 = [[0.8],[0.3]]
x4 = [[0.7],[0.1]]
x5 = [[0.5],[0.1]]
x6 = [[0.2],[0.1]]
x7 = [[0.2],[0.3]]
x8 = [[0.2],[0.7]]
c = x0
delta_x = np.subtract(x8, x0)

p_dot = - R_rc @ (np.linalg.pinv(J_image_n(c)) @ delta_x) # ---> this is the correct way

## ===================================== UR5E JACOBIAN TESTING ========================================================
p_dot0 = ([0, 0, 0, 0, 0, 0])
p_dot1 = ([0.2, 0, 0, 0, 0, 0])
delta_p = np.subtract(p_dot1, p_dot0)
q_dot = np.linalg.pinv(J_r(p_dot0)) @ delta_p
print("q_dot", q_dot)


# ## =================== INTELREALSENSE CAMERA INTRINSIC PARAMETERS =========================
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth)
# config.enable_stream(rs.stream.color)
# pipeline.start(config)
#
# # Get the depth profile and extract intrinsics
# depth_profile = pipeline.get_active_profile().get_stream(rs.stream.depth)
# depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
#
# # Get the color profile and extract intrinsics
# color_profile = pipeline.get_active_profile().get_stream(rs.stream.color)
# color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
#
# # Depth intrinsics
# print("Depth Intrinsics:")
# print(f"Width: {depth_intrinsics.width}")
# print(f"Height: {depth_intrinsics.height}")
# print(f"PPX (Principal Point X): {depth_intrinsics.ppx}")
# print(f"PPY (Principal Point Y): {depth_intrinsics.ppy}")
# print(f"FX (Focal Length X): {depth_intrinsics.fx}")
# print(f"FY (Focal Length Y): {depth_intrinsics.fy}")
# print(f"Distortion Model: {depth_intrinsics.model}")
# print(f"Distortion Coefficients: {depth_intrinsics.coeffs}")
#
# # Color intrinsics
# print("\nColor Intrinsics:")
# print(f"Width: {color_intrinsics.width}")
# print(f"Height: {color_intrinsics.height}")
# print(f"PPX (Principal Point X): {color_intrinsics.ppx}")
# print(f"PPY (Principal Point Y): {color_intrinsics.ppy}")
# print(f"FX (Focal Length X): {color_intrinsics.fx}")
# print(f"FY (Focal Length Y): {color_intrinsics.fy}")
# print(f"Distortion Model: {color_intrinsics.model}")
# print(f"Distortion Coefficients: {color_intrinsics.coeffs}")
#
# pipeline.stop()





















