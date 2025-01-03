"""
﷽
by @anbarsanti
"""

import numpy as np
import math
import torch
from matplotlib.path import Path
import cv2
from shapely.geometry import Polygon, Point, LineString
from supervision.tracker.byte_tracker.core import detections2boxes
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
import csv
import supervision as sv

from YOLOv11.controller import detected_box


## ===================================================================================
## ====================== TRACKING FUNCTION =========================
## ===================================================================================

def track_from_video(VIDEO_PATH, model):
    '''
    DETECT FROM VIDEO
    '''
    # Create a VideoCapture and ImageCapture object
    cap = cv2.VideoCapture(VIDEO_PATH)
    image = cv2.imread(IMAGE_PATH)
    aq
    
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
        # results = model(frame)
        results = model.track(frame, stream=True, show=True, persist=True,
                              tracker='bytetrack.yaml')  # Tracking with byteTrack
        
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
    
    return annotated_frame


def track_from_webcam(model, OBB=True):
    '''
    YOLO TRACK FROM WEB CAMERA STREAMING
    Return:
        detected_box in normalized cxyxyxyxy format (if OBB is true) or normalized cxyxyxy format (if OBB is false)
    '''
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
            for r in results:
                annotated_frame = r.plot()
                
                # Results Documentation:
                # https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results
                
                if OBB == True:
                    # Data Extraction from object tracking with OBB format
                    cls = r.obb.cls  # only applied in YOLO OBB model
                    xyxyxyxyn = r.obb.xyxyxyxyn  # Normalized [x1, y1, x2, y2, x3, y3, x4, y4] OBBs. only applied in YOLO OBB model
                    len_cls = len(cls)
                    for i in range(len_cls):
                        xyxyxyxyn_flatten = (np.array((xyxyxyxyn[i].tolist())).reshape(1, 8).tolist())[0]  # Flatten the xyxyxyxy
                        detected_box = [*[(cls[i].tolist())], *(xyxyxyxyn_flatten)]  # Append class with its OBB
                        yield detected_box
                       
                else:  # HBB
                    # Data Extraction from object tracking with HBB format
                    cls = r.boxes.cls  # Class labels for each HBB box. can't be applied in OBB
                    xyxyn = r.boxes.xyxyn  # Normalized [x1, y1, x2, y2] horizontal boxes relative to orig_shape. can't be applied in OBB
                    len_cls = len(cls)
                    for i in range(len_cls):
                        detected_box = [*[(cls[i].tolist())], *(xyxyn[i].tolist())]  # Append class with its HBB
                        yield detected_box
                
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
    
    return detected_box


def track_from_intelrealsense(model, OBB=True):
    '''
    YOLO DETECT FROM INTELREALSENSE CAMERA STREAMING
    Return:
        detected_box in normalized cxyxyxyxy format (if OBB is true) or normalized cxyxyxy format (if OBB is false)
    '''
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
    
    while True:
        frame = pipe.wait_for_frames()
        color_frame = frame.get_color_frame()
        
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        
        # # Show Images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("Bismillah", color_image)
        
        # Run YOLOv8 OBB inference on the frame
        results = model.track(color_image, stream=True, show=True, persist=True,
                              tracker='bytetrack.yaml')  # Tracking with byteTrack
        
        # Process and visualize the results of Object Tracking with YOLO (BUGS IN THIS PART)
        for r in results:
            annotated_frame = r.plot()
            
            if OBB==True:
                # Data Extraction from object tracking with OBB format
                cls = r.obb.cls  # only applied in YOLO OBB model
                xyxyxyxyn = r.obb.xyxyxyxyn  # Normalized [x1, y1, x2, y2, x3, y3, x4, y4] OBBs. only applied in YOLO OBB model
                len_cls = len(cls)
                for i in range(len_cls):
                    xyxyxyxyn_flatten = (np.array((xyxyxyxyn[i].tolist())).reshape(1, 8).tolist())[0]  # Flatten the xyxyxyxy
                    detected_box = [*[(cls[i].tolist())], *(xyxyxyxyn_flatten)]  # Append class with its OBB
                    yield detected_box
                    
            else:  # HBB
                # Data Extraction from object tracking with HBB format
                cls = r.boxes.cls  # Class labels for each HBB box. can't be applied in OBB
                xyxyn = r.boxes.xyxyn  # Normalized [x1, y1, x2, y2] horizontal boxes relative to orig_shape. can't be applied in OBB
                len_cls = len(cls)
                for i in range(len_cls):
                    detected_box = [*[(cls[i].tolist())], *(xyxyn[i].tolist())]  # Append class with its HBB
                    yield detected_box

            
            # Display the annotated frame
            cv2.imshow("YOLOv11 OBB Inference - Realsense Camera", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Stop Streaming
    pipe.stop()
    cv2.destroyAllWindows()
    
    return detected_box


## ===================================================================================
## ============================= YOLO HBB AND OBB FORMAT ================================

# YOLO OBB format is class_index x1 y1 x2 y2 x3 y3 x4 y4
# x1, y1: Coordinates of the first corner point
# x2, y2: Coordinates of the second corner point
# x3, y3: Coordinates of the third corner point
#  x4, y4: Coordinates of the fourth corner point
# All coordination are normalized to values between 0 and 1 relative to image dimensions (allows the annotation to be resolution dependent)
# YOLO v11 OBB format is already in polygon format

# YOLO HBB format is class_index x_center y_center width height


## ===================================================================================
## ====================== ARE BOX A AND BOX B INTERSECTING? =========================
## ====================== SEPARATION AXIS THEOREM ====================================
## ===================================================================================
# Useful sources:
# https://programmerart.weebly.com/separating-axis-theorem.html

def convert_HBB_to_vertices(hbb):
    """
    Convert Horizontal Bounding Boxes (HBB) from [class_id, x_center, y_center, width, height] to [[x1,y1],[x2,y2]]
    Where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    Args:
        box = input box of shape [1,5] with [class_id, x_center, y_center, width, height]
    Returns:
        Converted data in [[x1,y1],[x2,y2]] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    """
    x1y1 = [(hbb[1]-hbb[3]/2), (hbb[2]-hbb[4]/2)]
    x2y2 = [(hbb[1]+hbb[3]/2), (hbb[2]+hbb[4]/2)]
    return [x1y1, x2y2]

def convert_OBB_to_vertices(obb):
    """
    Convert Oriented Bounding Boxes (OBB) from [label, x1, y1, x2, y2, x3, y3, x4, y4] to [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    Args:
        box = input box of shape [1,9]: class_index x1 y1 x2 y2 x3 y3 x4 y4
    Returns:
        Converted data in [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    vertices = [[obb[i], obb[i + 1]] for i in range(1, len(obb), 2)]
    return vertices

def edges_of(vertices):
    """
    Return the vectors of the edges of the polygon
    Args:
        the vertices of the polygon
    Returns:
        the edges of the polygon
    """
    edges = []
    N = len(vertices)
    for i in range(N):
        edge = np.subtract(vertices[(i+1) % N],vertices[i]).tolist()
        edges.append(edge)
    return edges

def is_OBB_intersect(obbA, obbB):
    """
    Determining if two OBBs are intersecting using Separating Axis Theorem
    Args: two boxes with format  [label, x1, y1, x2, y2, x3, y3, x4, y4]
    Returns: True if two boxes are overlapping
    """
    def orthogonal(v):
        # Return a 90 degree clockwise rotation of the vector v
        return np.array([-v[1], v[0]])

    def project(vertices, axis):
        # project vertices [[1,2], [3,4], [5,6], [7,8]] onto axis [x,y]
        dots = [np.dot(vertex, axis) for vertex in vertices] / np.linalg.norm(axis)
        return [min(dots), max(dots)]

    edges = edges_of(convert_OBB_to_vertices(obbA)) + edges_of(convert_OBB_to_vertices(obbB))
    axes = [orthogonal(edge) for edge in edges]

    for axis in axes:
        projectionA = project(convert_OBB_to_vertices(obbA), axis)
        projectionB = project(convert_OBB_to_vertices(obbB), axis)
        if not (projectionA[0] < projectionB[1]) and (projectionB[0] < projectionA[1]):
            return False # box A and box B not overlapping, separating axis found
    return True # box A and box B is overlapping, no separating axis found

## ==============================================================================================================
## ====================== FIND INTERSECTION AREA IF TWO OBBS ARE OVERLAPPING ====================================
## ==============================================================================================================
## Other references using OpenCV: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9

def intersection_area_OBB_shapely(obbA, obbB):
    # Compute the intersection area using shapely library
    if is_OBB_intersect(obbA, obbB):
        polyA = Polygon(convert_OBB_to_vertices(obbA))
        polyB = Polygon(convert_OBB_to_vertices(obbB))
        return polyA.intersection(polyB).area
    else:
        return None

def cxyxyxyxy2xywhr(OBB):
    """
    Converts bounding box with format [class x1 y1 x2 y2 x3 y3 x4 y4] to [cx, cy, w, h, rotation] format
    Args: bounding box with format [class x1 y1 x2 y2 x3 y3 x4 y4]
    Return: [cx, cy, w, h, rotation] format, rotation is returned in radians from 0 to pi/2, as a column vector with shape (1,5)
    """

    # Calculate center
    cx = np.mean(np.array([OBB[1], OBB[3], OBB[5], OBB[7]]))
    cy = np.mean(np.array([OBB[2], OBB[4], OBB[6], OBB[8]]))

    # Calculate width and height
    vertices = convert_OBB_to_vertices(OBB)
    w = np.mean(np.array([math.dist(vertices[0],vertices[1]), math.dist(vertices[2],vertices[3])]))
    h = np.mean(np.array([math.dist(vertices[1],vertices[2]), math.dist(vertices[3],vertices[0])]))

    # Calculate (simplify) angle
    dx = vertices[1][0] - vertices[0][0]
    dy = vertices[1][1] - vertices[0][1]
    rot = math.degrees(math.atan2(dy, dx))/180*np.pi

    # # Compare with cv2.minAreaRect
    # points = np.array(box[1:]).reshape((-1, 2)).astype(np.float32)
    # rect = cv2.minAreaRect(points)

    return np.array([cx, cy, w, h, rot]).reshape(-1,1)

def cxyxyxyxy2xyxyxy(OBB):
    """
    Converts horizontal bounding box with format [class x1 y1 x2 ] to [x1 y1 x2 y2 x3 y3] as image feature vector
    x1 y1 and x2 y2 represent the midpoint on the adjacent two sides of the rotating bounding box
    x3 y3 denotes the center of rotation bounding box
    Args: bounding box with format [class x1 y1 x2 y2 x3 y3 x4 y4]
    Return: image feature vector with [x1 y1 x2 y2 x3 y3] as a column vector with shape (5,1)
    """

    # Calculate center
    x3 = np.mean(np.array([OBB[1], OBB[3], OBB[5], OBB[7]]))
    y3 = np.mean(np.array([OBB[2], OBB[4], OBB[6], OBB[8]]))

    vertices = convert_OBB_to_vertices(OBB)
    x1, y1 = np.mean([vertices[0], vertices[1]], axis=0).tolist()
    x2, y2 = np.mean([vertices[0], vertices[3]], axis=0).tolist()

    return np.array([x1, y1, x2, y2, x3, y3]).reshape(-1,1)

def cxywh2xyxyxy(HBB):
    """
    Converts bounding box with format [class x_center y_center width height] to [x1 y1 x2 y2 x3 y3] as image feature vector
    x1 y1 and x2 y2 represent the midpoint on the adjacent two sides of the rotating bounding box
    x3 y3 denotes the center of horizontal bounding box
    Args:
        horizontal bounding box with format [class x_center y_center width height]
    Return:
        image feature vector with [x1 y1 x2 y2 x3 y3]
    """
    # Calculate
    x1 = HBB[1] - (HBB[3]/2)
    y1 = HBB[2]
    x2 = HBB[1]
    y2 = HBB[2] - (HBB[4]/2)
    x3 = HBB[1]
    y3 = HBB[2]
    
    return np.array([x1, y1, x2, y2, x3, y3]).reshape(-1,1)
    
def line_intersection(line1, line2):
    """
    Finding the intersection point between two lines
    Args: lines that have format [[x1, y1], [x2, y2]]
    Return: [cx, cy, w, h, rotation] format, rotation is returned in radians from 0 to pi/2
    """
    x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
    x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]

    # Calculate the denominator
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

    # Check if lines are parallel
    if denom == 0:
        return None

    # Calculate the intersection point
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    # Check if the intersection point lies on both line segments
    if ua < 0 or ua > 1 or ub < 0 or ub > 1:
        return None

    # Calculate the (x, y) coordinates of the intersection point
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)

    return [x, y]

def sort_points_clockwise(points):
    """
    Sorting some points in convex polygon in clockwise order
    Args: list of array of points with format [[x, y],[x, y],[x, y],[x, y],...,[x, y]]
    Return: sorted list of array with format [[x, y],[x, y],[x, y],...,[x, y]] in clockwise order
    """
    # Convert to numpy array
    points = np.array(points)
    
    # Calculate centroid
    centroid = np.mean(points, axis=0)
    
    # Compute angles
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    
    # Sort points based on angles
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]

    return sorted_points

def intersection_points_OBB_diy(boxA, boxB):
    """
    Implementation of Polygon Clipping for intersection area computation
    Here I am not implementing Sutherland Polygon Clipping since it does not cover the edge that intersect the other edge twice
    Args: two OBBs with format  [class, x1, y1, x2, y2, x3, y3, x4, y4] with shape (1,9)
    Returns: intersection points with shape (1,p)
    """
    subject_vertices = convert_OBB_to_vertices(boxA)
    clipper_vertices = convert_OBB_to_vertices(boxB)
    subject_polygon = Path(subject_vertices) # box A acts as subject polygon, the polygon to be clipped
    clipper_polygon = Path(clipper_vertices) # box B acts as clipper polygon

    intersection_points = []

    # Identify vertices from each polygon that are contained within the other polygon.
    for i in range(len(clipper_vertices)):
        if subject_polygon.contains_point(clipper_vertices[i]):
            intersection_points.append(clipper_vertices[i]) # These vertices will be part of the intersection area.

    for i in range(len(subject_vertices)):
        if clipper_polygon.contains_point(subject_vertices[i]):
            intersection_points.append(subject_vertices[i]) # These vertices will be part of the intersection area.

    # Edge Intersection Points become new vertices in the intersection polygon
    len_clipper = len(clipper_vertices)
    len_subject = len(subject_vertices)
    for i in range(len_clipper):
        for j in range(len_subject):
            line1 = ([clipper_vertices[i], clipper_vertices[(i+1) % len_clipper]])
            line2 = ([subject_vertices[j], subject_vertices[(j+1) % len_subject]])
            point = line_intersection(line1, line2)

            if point is not None:
                if not any (p == point for p in intersection_points):
                    intersection_points.append(point)

    # Sorting the points in clockwise order
    sorted_points = sort_points_clockwise(intersection_points)

    return sorted_points

def intersection_area_OBB_diy(boxA, boxB):
    """
    Compute the Area using Triangle Formula byMauren Abreu de Souza
    Args: two OBBs with format  [class, x1, y1, x2, y2, x3, y3, x4, y4] with shape (1,9)
    Returns: intersection area with shape (1,1)
    """
    intersection_points = intersection_points_OBB_diy(boxA, boxB)
    len_points = len(intersection_points)
    area = 0.0
    for i in range(len_points):
        j = (i + 1) % len_points
        area += intersection_points[i][0] * intersection_points[j][1]
        area -= intersection_points[j][0] * intersection_points[i][1]

    area = abs(area) / 2.0

    return area

def intersection_area_HBB(boxA, boxB):
    """
    Compute the Intersection Area of two Horizontal Bounding Boxes (HBB)
    Args: two HBBs with format  [class, x1, y1, x2, y2, x3, y3, x4, y4] with shape (1,9)
    Returns: intersection area with shape (1,1)
    """
    verticesA = convert_HBB_to_vertices(boxA)
    verticesB = convert_HBB_to_vertices(boxB)

    def is_HBB_intersect(verticesA, verticesB):
        return (verticesA[0][0] < verticesB[1][0] and
                verticesA[1][0] > verticesB[0][0] and # boxA.minX <= boxB.maxX and boxA.maxX >= boxB.minX
                verticesA[0][1] < verticesB[1][1] and
                verticesA[1][1] > verticesB[0][1]) # (boxA.minY <= boxB.maxY) and (boxA.maxY >= boxB.minY)

    if is_HBB_intersect(verticesA, verticesB):
        # Calculate the coordinates of intersection rectangle
        x_left = max(verticesA[0][0], verticesB[0][0])
        y_top = max(verticesA[0][1], verticesB[0][1])
        x_right = min(verticesA[1][0], verticesB[1][0])
        y_bottom = min(verticesA[1][1], verticesB[1][1])

        area = (x_right-x_left)*(y_bottom-y_top)

    else:
        area = 0

    return area


## ==============================================================================================================
## ============================================ DEFINE THE JACOBIAN MATRICES ====================================
## ==============================================================================================================

def J_alpha(intersection_points):
    """
    Return J_alpha, the Jacobian matrix that maps the area from intersection points to image space.
    Args:
        the vertices of the polygon
    Returns:
        J_alpha
    """
    len_points = len(intersection_points)
    jacobian = []
    for i in range(len_points):
        j = (i + 1) % len_points
        k = (i + len_points) % len_points
        jacobian.append(intersection_points[j][1] - intersection_points[k][1])
        jacobian.append(intersection_points[k][0] - intersection_points[j][0])
    
    # Compute the determinant to check singularity
    J_alpha = np.array(jacobian)
    # determinant = np.linalg.det(J_alpha)
    
    return J_alpha

def J_a(intersection_points, depth):
    """
	 Construct the Jacobian matrix that maps intersection points in image space to linear velocity and angular velocity to cartesian space.
	 Args:
		  a list of intersection points
	 Returns:
		  J_I (8x3 matrix)
	 """
    # Precompute general terms
    f_x = 618.072
    f_y = 618.201
    depth = 1000.0
    len_points = len(intersection_points)
    p = intersection_points
    jacobian = []
    
    # Define the Jacobian
    for i in range(len_points):
        j = [f_x / depth, 0, -p[i][0] / depth, -p[i][0] * p[i][1] / f_x, (f_x * f_x + p[i][0] * p[i][0]) / f_x,
             -p[i][1]]
        k = [0, f_y / depth, -p[i][1] / depth, -(f_y * f_y + p[i][1] * p[i][1]) / f_y, p[i][0] * p[i][1] / f_y, p[i][0]]
        jacobian.extend([[j], [k]])
    
    # Compute the determinant to check singularity
    J_a = np.array(jacobian)
    # determinant = np.linalg.det(J_a)
    
    return J_a

def J_I(p):
    """
	 Return the Jacobian matrix that maps 3 points in image space to linear velocity and angular velocity to cartesian space.
	 Args:
		  image feature points [x1, y1, x2, y2, x3, y3].T in a column vector format
	 Returns:
		  J_I (6x6 matrix)
	 """
    # Precompute general terms
    f_x = 618.072
    f_y = 618.201
    depth = 1000.0
    xc = p[0,0]
    yc = p[1,0]
    x1 = p[2,0]
    y1 = p[3,0]
    x2 = p[4,0]
    y2 = p[5,0]
    
    image_jacobian = [
        [-f_x / depth, 0, xc / depth, xc * yc / f_x, -(f_x * f_x + xc * xc) / f_x, yc],
        [0, -f_y / depth, yc / depth, (f_y * f_y + yc * yc) / f_y, -xc * yc / f_y, -xc],
        [-f_x / depth, 0, x1 / depth, x1 * y1 / f_x, -(f_x * f_x + x1 * x1) / f_x, y1],
        [0, -f_y / depth, y1 / depth, (f_y * f_y + y1 * y1) / f_y, -x1 * y1 / f_y, -x1],
        [-f_x / depth, 0, x2 / depth, x2 * y2 / f_x, -(f_x * f_x + x2 * x2) / f_x, y2],
        [0, -f_y / depth, y2 / depth, (f_y * f_y + y2 * y2) / f_y, -x2 * y2 / f_y, -x2],
    ]
    return np.array(image_jacobian)

def J_o(p):
    """
	 Construct the Jacobian matrix that maps yolo points to image feature points (3 points)
	 Args:
		  image feature points [x1, y1, x2, y2, x3, y3].T in a column vector format
		  p_1 and p_2 represent the midpoint on the adjacent two sides of the rotating bounding box
		  p_3 denotes the center of rotation bounding box
	 Returns:
		 The jacobian matrix J_o (5x4 matrix)
	 """
    # Precompute common terms
    jacobian = np.zeros((5, 6))
    x1 = p[0,0]
    y1 = p[1,0]
    x2 = p[2,0]
    y2 = p[3,0]
    x3 = p[4,0]
    y3 = p[5,0]
    delta_x1 = x3 - x1;
    delta_y1 = y1 - y3;
    delta_y2 = y3 - y2;
    delta_x2 = x3 - x2;
    theta = np.arctan2(delta_y1, delta_x1)
    denom1 = delta_x1 ** 2 + delta_y1 ** 2;
    sqrt_denom1 = np.sqrt(denom1)
    denom2 = delta_x2 ** 2 + delta_y2 ** 2;
    sqrt_denom2 = np.sqrt(denom2)
    
    # Ensure inputs are not zero to avoid division by zero
    assert denom1 > 0, "Invalid input: delta_x1^2 + delta_y1^2 must be > 0"
    assert denom2 > 0, "Invalid input: delta_x2^2 + delta_y2^2 must be > 0"
    
    # Define the jacobian
    jacobian[0, :] = [0, 0, 0, 0, 1, 0]
    jacobian[1, :] = [0, 0, 0, 0, 0, 1]
    jacobian[2, :] = [-2 * delta_x1 / sqrt_denom1, 2 * delta_y1 / sqrt_denom1, 0, 0, 2 * delta_x1 / sqrt_denom1,
                      -2 * delta_y1 / sqrt_denom1]
    jacobian[3, :] = [0, 0, -2 * delta_x2 / sqrt_denom2, -2 * delta_y2 / sqrt_denom2, 2 * delta_x2 / sqrt_denom2,
                      2 * delta_y2 / sqrt_denom2]
    jacobian[4, :] = [delta_y1 / denom1, delta_x1 / denom1, 0, 0, -delta_y1 / denom1, -delta_x1 / denom1]
    
    # Compute the determinant to check singularity
    # determinant = np.linalg.det(jacobian)
    
    return jacobian

def numerical_J_o(p, epsilon=1e-8):
    """
	 Use numerical approximation to verify the analytical Jacobian matrix J_o
	 Args:
		 image feature points [x1, y1, x2, y2, x3, y3].T in a column vector format
		 p_1 and p_2 represent the midpoint on the adjacent two sides of the rotating bounding box
		 p_3 denotes the center of rotation bounding box
	 Returns: assertion results
	 """
    
    # Define the Sigma function
    def sigma(p):
        # Precompute common terms
        x1 = p[0, 0]
        y1 = p[1, 0]
        x2 = p[2, 0]
        y2 = p[3, 0]
        x3 = p[4, 0]
        y3 = p[5, 0]
        delta_x1 = x3 - x1;
        delta_y1 = y1 - y3;
        delta_y2 = y3 - y2;
        delta_x2 = x3 - x2;
        theta = np.arctan2(delta_y1, delta_x1)
        denom1 = delta_x1 ** 2 + delta_y1 ** 2;
        sqrt_denom1 = np.sqrt(denom1)
        denom2 = delta_x2 ** 2 + delta_y2 ** 2;
        sqrt_denom2 = np.sqrt(denom2)
        return [x3, y3, 2 * sqrt_denom1, 2 * sqrt_denom2, theta]
    
    # Build the Jacobian matrix using numerical approach
    n = len(p)
    m = len(sigma(p))
    jacobian = np.zeros((m, n))
    
    for i in range(n):
        p_plus = p.copy()
        p_plus[i] += epsilon
        p_minus = p.copy()
        p_minus[i] -= epsilon
        jacobian[:, i] = np.subtract(sigma(p_plus), sigma(p_minus)) / (2 * epsilon)
        # print(jacobian.shape)
        # print(jacobian)
    
    # Test
    # The function below compares two arrays, numerical_jacobian and analytical_jacobian, element wise
    # Checking if they are equal within the specified tolerance atol
    # numerical_jacobian = sigma(p)
    # analytical_jacobian = J_o(p)
    # return np.testing.assert_allclose(numerical_jacobian, analytical_jacobian, rtol = 1e-04, atol=1e-05)
    return jacobian

def J_r_linear(q):
    """
	 Compute the 3x6 linear part of the Jacobian matrix for UR5e.
	 Param:
		 q: numpy array of joint angles [[q1], [q2], [q3], [q4], [q5], [q6]] in column vector
	 Return:
		 3x6 Jacobian matrix (linear part only)
	 """
    
    # Get the full 6x6 Jacobian
    J = J_r(q)
    
    # Extract the 3x6 linear velocity part: from the top 3 rows, and all columnts
    J_linear = J[:3, :]
    
    # Compute the determinant to check singularity
    determinant = np.linalg.det(J_linear)
    
    return J_linear

def J_r_angular(q):
    """
	 Compute the 3x6 angular part of the Jacobian matrix for UR5e.
	 Param:
		 q: numpy array of joint angles [[q1], [q2], [q3], [q4], [q5], [q6]] in column vector
	 Return:
		 3x6 Jacobian matrix (angular part only)
	 """
    
    # Get the full 6x6 Jacobian
    J = J_r(q)
    
    # Extract the 3x6 linear velocity part:last three rows in the matrix
    J_angular = J[-3:, :]
    
    # Compute the determinant to check singularity
    determinant = np.linalg.det(J_angular)
    
    return J_angular

def J_r(q):
    """
	 Construct the Jacobian matrix that maps velocity in cartesian space to joint velocity in UR5e joint space
	 Based on inverse kinematics of UR5e robotic arm and avoid the singularities that might be happened
	 Source:
		 Singularity Analysis and Complete Methods to Compute the Inverse Kinematics for a 6-DOF UR/TM-Type Robot
		 Jessice Villalobos
	 Args:
		 q: numpy array of joint angles [[q1], [q2], [q3], [q4], [q5], [q6]] in column vector
	 Returns:
		 Jacobian matrix J_r (6x6 matrix), including linear and angular velocity parts
	 """
    # Precompute & Predefine some terms
    pi = 3.1415926535
    jacobian = np.zeros((6, 6))
    d1 = 0.0892  # 0.1625
    d4 = 0.1093  # 0.1333
    d5 = 0.09475  # 0.0997
    d6 = 0.0825  # 0.0996
    a2 = 0.425
    a3 = 0.392
    q1 = q[0,0]
    q2 = q[1,0]
    q3 = q[2,0]
    q4 = q[3,0]
    q5 = q[4,0]
    q6 = q[5,0]
    c1 = math.cos(q1)
    c2 = math.cos(q2)
    c3 = math.cos(q3)
    c4 = math.cos(q4)
    c5 = math.cos(q5)
    c6 = math.cos(q6)
    c23 = math.cos(q2 + q3)
    c234 = math.cos(q2 + q3 + q4)
    s1 = math.sin(q1)
    s2 = math.sin(q2)
    s3 = math.sin(q3)
    s4 = math.sin(q4)
    s5 = math.sin(q5)
    s6 = math.sin(q6)
    s23 = math.sin(q2 + q3)
    s234 = math.sin(q2 + q3 + q4)
    r13 = -(c1 * c234 * s5) + (c5 * s1)
    r23 = -(c234 * s1 * s5) - (c1 * c5)
    r33 = - (s234 * s5)
    px = (r13 * d6) + (c1 * ((s234 * d5) + (c23 * a3) + (c2 * a2))) + (s1 * d4)
    py = (r23 * d6) + (s1 * ((s234 * d5) + (c23 * a3) + (c2 * a2))) - (c1 * d4)
    pz = (r33 * d6) - (c234 * d5) + (s23 * a3) + (s2 * a2) + d1
    
    # Define the Jacobian Matrix
    jacobian[0, 0] = -py
    jacobian[0, 1] = -c1 * (pz - d1)
    jacobian[0, 2] = c1 * (s234 * s5 * d6 + (c234 * d5) - (s23 * a3))
    jacobian[0, 3] = c1 * ((s234 * s5 * d6) + (c234 * d5))
    jacobian[0, 4] = -d6 * ((s1 * s5) + (c1 * c234 * c5))
    jacobian[0, 5] = 0
    jacobian[1, 0] = px
    jacobian[1, 1] = -s1 * (pz - d1)
    jacobian[1, 2] = s1 * ((s234 * s5 * d6) + (c234 * d5) - (s23 * a3))
    jacobian[1, 3] = s1 * ((s234 * s5 * d6) + (c234 * d5))
    jacobian[1, 4] = d6 * ((c1 * s5) - (c234 * c5 * s1))
    jacobian[1, 5] = 0
    jacobian[2, 0] = 0
    jacobian[2, 1] = (s1 * py) + (c1 * px)
    jacobian[2, 2] = -(c234 * s5 * d6) + (s234 * d5) + (c23 * a3)
    jacobian[2, 3] = -(c234 * s5 * d6) + (s234 * d5)
    jacobian[2, 4] = -c5 * s234 * d6
    jacobian[2, 5] = 0
    
    jacobian[3, 0] = 0
    jacobian[3, 1] = s1
    jacobian[3, 2] = s1
    jacobian[3, 3] = s1
    jacobian[3, 4] = c1 * s234
    jacobian[3, 5] = r13
    jacobian[4, 0] = 0
    jacobian[4, 1] = -c1
    jacobian[4, 2] = -c1
    jacobian[4, 3] = -c1
    jacobian[4, 4] = s1 * s234
    jacobian[4, 5] = r23
    jacobian[5, 0] = 1
    jacobian[5, 1] = 0
    jacobian[5, 2] = 0
    jacobian[5, 3] = 0
    jacobian[5, 4] = -c234
    jacobian[5, 5] = r33
    
    # Revise (still needs to be checked)
    jacobian[0, 0] = -jacobian[0, 0]
    jacobian[0, 1] = -jacobian[0, 1]
    jacobian[0, 2] = -jacobian[0, 2]
    jacobian[1, 0] = -jacobian[1, 0]
    jacobian[1, 1] = -jacobian[1, 1]
    jacobian[1, 2] = -jacobian[0, 2]
    jacobian[2, 0] = -jacobian[2, 0]
    jacobian[2, 1] = -jacobian[2, 1]
    jacobian[2, 2] = -jacobian[0, 2]
    
    # Compute the determinant to check singularity
    determinant = np.linalg.det(jacobian)
    
    # Return the Jacobian Matrix
    return jacobian

def r2r_control(reaching_box, desired_box, actual_q, OBB=True):
    """
	 Construct the controller that consists of 4 steps: (1) Reaching, (2) Overlapping,
	 (3)Scaling & Screwing, and finally, (4) Desired Overlapping.
	 Source: https://www.rosroboticslearning.com/jacobian
	 Args:
		 reaching_box: the reaching target in YOLO OBB format
		 desired_box: the desired target in YOLO OBB format
		 actual_q: numpy array of joint angles [[q1], [q2], [q3], [q4], [q5], [q6]] in column vector
		 YOLO HBB format is class_index x_center y_center width height
	 Returns:
		 q_dot, the joint velocity of UR5e robotic arm
	 """
    # -------------------------- REACHING STATE -----------------------------------
    # Precompute & Predefine some terms
    e_cx = 0.002
    e_cy = 0.003
    k_cx = 0.0001
    k_cy = 0.0001
    k = 1
    P_r = 0.002
    n = 2
    speed = 1e-06
    
    # Conversion for OBB or HBB to xywhr format
    if OBB:
        r_box = cxyxyxyxy2xywhr(reaching_box)
        d_box = cxyxyxyxy2xywhr(desired_box)
        p_r_box = cxyxyxyxy2xyxyxy(reaching_box) # convert to xyxyxy format (image feature points) for OBB reaching box
    else:  # HBB
        r_box = np.array([reaching_box[1:5], 0]).reshape(-1,1)
        d_box = np.array([desired_box[1:5], 0]).reshape(-1,1)
        p_r_box = cxywh2xyxyxy(reaching_box) # convert to xyxyxy format (image feature points) for HBB reaching box
    
    # Reaching State --> Objective Function
    f_cx = abs(r_box[0,0] - d_box[0,0]) ** 2 - e_cx ** 2
    f_cy = abs(r_box[1,0] - d_box[1,0]) ** 2 - e_cy ** 2
    
    # Reaching State --> Energy Function
    P_R = (k_cx / n) * (max(0, f_cx) ** n) + (k_cy / n) * (max(0, f_cy) ** n) + P_r
    
    # Differentiation of P_R
    epsilon_R = np.array([(2 * k_cx / (n ** 2)) * ((max(0, f_cx)) ** (n - 1)) * (r_box[0,0] - d_box[0,0]),
               (2 * k_cy / (n ** 2)) * ((max(0, f_cy)) ** (n - 1)) * (r_box[1,0] - d_box[1,0]), 0, 0, 0])
    print("epsilon_R: ", epsilon_R)
    
    # Compute the full Jacobian matrix for current joint position values (actual_q) and position of reaching box
    J_reaching = (J_o(p_r_box)) @ (J_I(p_r_box)) @ (J_r(actual_q)) # Step 3
    print("J_reaching: ", J_reaching)
    
    # Compute the pseudo inverse of the Jacobian matrix using the Moore-Penrose matrix inversion
    J_reaching_pinv = np.linalg.pinv(J_reaching) # Step 4
    print("J_reaching_pinv: ", J_reaching_pinv)
    
    # Delta_Gamma or Gamma dot
    Gamma_dot = speed*(d_box - r_box) # Step 1 & 2
    print("Gamma_dot: ", Gamma_dot)
    
    # The Controller
    q_dot = J_reaching_pinv @ Gamma_dot # Step 5. Using np.dot has similar value with using @
    print("q_dot: ", q_dot)
    
    return q_dot