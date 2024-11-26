# Bismillah
from supervision.tracker.byte_tracker.core import detections2boxes
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
import csv

import supervision as sv

# Load a model
model = YOLO("model/yolo11-hbb-toy-11-26-2.pt") # toys for HBB object tracking
# model = YOLO("model/yolo11n.pt") # object tracking with HBB

# ===================================================================================== ##
# ================================= DETECT FROM VIDEO ================================= ##
# ===================================================================================== ##

# # Set up video path and image path
# VIDEO_PATH = "test/VID02.mp4"
# IMAGE_PATH = "test/P1.jpg"
#
# # Create a VideoCapture and ImageCapture object
# cap = cv2.VideoCapture(VIDEO_PATH)
# image = cv2.imread(IMAGE_PATH)aq
#
# # Get video properties
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
#
# # Create VideoWriter object for output
# output_path = "output_video.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Run YOLOv8 OBB inference on the frame
#     results = model(frame)
#     results = model.track(frame, stream=True, show=True, persist=True, tracker='bytetrack.yaml')  # Tracking with byteTrack
#
#     # Draw the detections on the frame
#     annotated_frame = results[0].plot()
#
#     # Write the frame to the output video
#     out.write(annotated_frame)
#
#     # Display the frame (optional)
#     cv2.imshow("YOLOv11 OBB Inference", annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# # =============================================================================== ##
# ## ====================== DETECT FROM WEB CAMERA STREAMING ====================== ##
# ===================================================================================== ##
#
# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or change to a specific camera index if needed
# 0 = web camera, 2 = depth camera

# Set the desired frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

boxes = []
scores = []

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run YOLOv8 OBB tracking on the frame. Tracking also can be used for OBB
        # persist = True --> to maintain track continuity between frames
        results = model.track(frame, stream=True, show=True, persist=True,
                              tracker='bytetrack.yaml')  # Tracking with byteTrack

        # Process, extract, and visualize the results
        for r in results:
            annotated_frame = r.plot()

            # Results Documentation:
            # https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results

            # # Print extracted data from object tracking with HBB format
            # cls = r.boxes.cls # Class labels for each HBB box. can't be applied in OBB
            # xyxyn = r.boxes.xyxyn # Normalized [x1, y1, x2, y2] horizontal boxes relative to orig_shape. can't be applied in OBB
            # len_cls = len(cls)
            # for i in range(len_cls):
            #     cxyxyn = [*[(cls[i].tolist())], *(xyxyn[i].tolist())] # Append class with its HBB
            #     # print(cxyxyn)
            #     with open('file_hbb.csv', 'a', newline='') as file:
            #         writer = csv.writer(file)
            #         writer.writerows([cxyxyn])

            # # Print extracted data from object tracking with OBB format
            # cls = r.obb.cls # only applied in YOLO OBB model
            # xyxyxyxyn = r.obb.xyxyxyxyn # Normalized [x1, y1, x2, y2, x3, y3, x4, y4] OBBs. only applied in YOLO OBB model
            # len_cls = len(cls)
            # for i in range(len_cls):
            #     xyxyxyxyn_flatten = (np.array((xyxyxyxyn[i].tolist())).reshape(1, 8).tolist())[0]  # Flatten the xyxyxyxy
            #     cxyxyxyxyn = [*[(cls[i].tolist())], *(xyxyxyxyn_flatten)]  # Append class with its HBB
            #     print(cxyxyxyxyn)
            #     with open('file_obb.csv', 'a', newline='') as file:
            #         writer = csv.writer(file)
            #         writer.writerows([cxyxyxyxyn])

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



## =============================================================================================== ##
## ====================== DETECT FROM INTELREALSENSE CAMERA STREAMING ====================== ##
## ================================================================================================
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
#     results = model.track(frame, stream=True) #, show=True, persist=True, tracker='bytetrack.yaml')  # Tracking with byteTrack
#
#     # Process and visualize the results
#     for r in results:
#         annotated_frame = r.plot()
#
#     # annotated_frame = results[0].plot()
#
#         # Display the annotated frame
#         cv2.imshow("YOLOv11 OBB Inference - Realsense Camera", annotated_frame)
#
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# pipe.stop()
