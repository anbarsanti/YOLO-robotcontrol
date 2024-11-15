# Bismillah
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2

import supervision as sv

# Load a model
model = YOLO("model/yolo11-obb-10-17-watercan.pt")

# ===================================================== ##
# ================= DETECT FROM VIDEO ================= ##

# Set up video path and image path
VIDEO_PATH = "test/VID02.mp4"
IMAGE_PATH = "test/P1.jpg"

# Create a VideoCapture and ImageCapture object
cap = cv2.VideoCapture(VIDEO_PATH)
image = cv2.imread(IMAGE_PATH)aq        

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object for output
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 OBB inference on the frame
    results = model(frame)

    # Draw the detections on the frame
    annotated_frame = results[0].plot()

    # Write the frame to the output video
    out.write(annotated_frame)

    # Display the frame (optional)
    cv2.imshow("YOLOv11 OBB Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# # =============================================================================== ##
# ## ====================== DETECT FROM WEB CAMERA STREAMING ====================== ##
#
# # Open the camera
# cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or change to a specific camera index if needed
# # 0 = web camera
# # 2 = depth camera
#
#
# # Set the desired frame width and height
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Run YOLOv8 OBB inference on the frame
#     results = model.track(frame, stream=True)
#
#     # Process and visualize the results
#     for r in results:
#         annotated_frame = r.plot()
#
#         # Display the annotated frame
#         cv2.imshow("YOLOv11 OBB Inference - Webcam", annotated_frame)
#
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# ## =============================================================================== ##
# ## ====================== DETECT FROM INTELREALSENSE CAMERA STREAMING ====================== ##
#
# pipe = rs.pipeline()
# cfg  = rs.config()
#
# cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
# # cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
#
# pipe.start(cfg)
#
# while True:
#     frame = pipe.wait_for_frames()
#     color_frame = frame.get_color_frame()
#     color_image = np.asanyarray(color_frame.get_data())
#
#     # Run YOLOv8 OBB inference on the frame
#     results = model.track(color_image, stream=True)
#
#     # Process and visualize the results
#     for r in results:
#         annotated_frame = r.plot()
#
#         # Display the annotated frame
#         cv2.imshow("YOLOv11 OBB Inference - Realsense Camera", annotated_frame)
#
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# pipe.stop()
