"""
﷽
author: @anbarsanti
"""

import numpy as np
import math
from matplotlib.path import Path
import cv2
from openpyxl.utils.units import dxa_to_cm
from shapely.geometry import Polygon, Point, LineString
import torch


# Useful sources:
# https://programmerart.weebly.com/separating-axis-theorem.html

# YOLO OBB format is class_index x1 y1 x2 y2 x3 y3 x4 y4
# x1, y1: Coordinates of the first corner point
# x2, y2: Coordinates of the second corner point
# x3, y3: Coordinates of the third corner point
#  x4, y4: Coordinates of the fourth corner point
# All coordination are normalized to values between 0 and 1 relative to image dimensions (allows the annotation to be resolution dependent)
# YOLO v11 OBB format is already in polygon format

## ===================================================================================
## ====================== ARE BOX A AND BOX B INTERSECTING? =========================
## ====================== SEPARATION AXIS THEOREM ====================================
## ===================================================================================

def convert_to_vertices(box):
    """
    Convert Oriented Bounding Boxes (OBB) from [label, x1, y1, x2, y2, x3, y3, x4, y4] to [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    Args:
        box = input box of shape [1,9]
    Returns:
        Converted data in [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    vertices = [[box[i], box[i + 1]] for i in range(1, len(box), 2)]
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

def is_intersect(boxA, boxB):
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

    edges = edges_of(convert_to_vertices(boxA)) + edges_of(convert_to_vertices(boxB))
    axes = [orthogonal(edge) for edge in edges]

    for axis in axes:
        projectionA = project(convert_to_vertices(boxA), axis)
        projectionB = project(convert_to_vertices(boxB), axis)
        if not (projectionA[0] < projectionB[1]) and (projectionB[0] < projectionA[1]):
            return False # box A and box B not overlapping, separating axis found
    return True # box A and box B is overlapping, no separating axis found

## ==============================================================================================================
## ====================== FIND INTERSECTION AREA IF TWO OBBS ARE OVERLAPPING ====================================
## ==============================================================================================================
## Other references using OpenCV: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9

def intersection_area_shapely(boxA, boxB):
    # Compute the intersection area using shapely library
    if is_intersect(boxA, boxB):
        polyA = Polygon(convert_to_vertices(boxA))
        polyB = Polygon(convert_to_vertices(boxB))
        return polyA.intersection(polyB).area
    else:
        return None

def cxyxyxyxy2xywhr(box):
    """
    Converts bounding box with format [class x1 y1 x2 y2 x3 y3 x4 y4] to [cx, cy, w, h, rotation] format
    Args: bounding box with format [class x1 y1 x2 y2 x3 y3 x4 y4]
    Return: [cx, cy, w, h, rotation] format, rotation is returned in radians from 0 to pi/2
    """

    # Calculate center
    cx = np.mean(np.array([box[1], box[3], box[5], box[7]]))
    cy = np.mean(np.array([box[2], box[4], box[6], box[8]]))

    # Calculate width and height
    vertices = convert_to_vertices(box)
    w = np.mean(np.array([math.dist(vertices[0],vertices[1]), math.dist(vertices[2],vertices[3])]))
    h = np.mean(np.array([math.dist(vertices[1],vertices[2]), math.dist(vertices[3],vertices[0])]))

    # Calculate (simplify) angle
    dx = vertices[1][0] - vertices[0][0]
    dy = vertices[1][1] - vertices[0][1]
    rot = math.degrees(math.atan2(dy, dx))/180*np.pi

    # # Compare with cv2.minAreaRect
    # points = np.array(box[1:]).reshape((-1, 2)).astype(np.float32)
    # rect = cv2.minAreaRect(points)

    return [cx, cy, w, h, rot]

def cxyxyxyxy2xyxyxy(box):
    """
    Converts bounding box with format [class x1 y1 x2 y2 x3 y3 x4 y4] to [x1 y1 x2 y2 x3 y3] as image feature vector
    x1 y1 and x2 y2 represent the midpoint on the adjacent two sides of the rotating bounding box
    x3 y3 denotes the center of rotation bounding box
    Args: bounding box with format [class x1 y1 x2 y2 x3 y3 x4 y4]
    Return: image feature vector with [x1 y1 x2 y2 x3 y3]
    """

    # Calculate center
    x3 = np.mean(np.array([box[1], box[3], box[5], box[7]]))
    y3 = np.mean(np.array([box[2], box[4], box[6], box[8]]))

    vertices = convert_to_vertices(box)
    x1, y1 = np.mean([vertices[0], vertices[1]], axis=0).tolist()
    x2, y2 = np.mean([vertices[0], vertices[3]], axis=0).tolist()

    return [x1, y1, x2, y2, x3, y3]

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
    # Convert points to numpy array
    points = np.array(points)

    # Step 1: Find the centroid
    centroid = np.mean(points, axis=0)

    # Step 2: Calculate angles
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Step 3: Sort points based on angles
    sorted_indices = np.argsort(-angles)  # Negative for clockwise order
    sorted_points = points[sorted_indices]

    return sorted_points.tolist()

def intersection_points_diy(boxA, boxB):
    """
    Implementation of Polygon Clipping for intersection area computation
    Here I am not implementing Sutherland Polygon Clipping since it does not cover the edge that intersect the other edge twice
    Args: two OBBs with format  [class, x1, y1, x2, y2, x3, y3, x4, y4] with shape (1,9)
    Returns: intersection points with shape (1,p)
    """
    subject_vertices = convert_to_vertices(boxA)
    clipper_vertices = convert_to_vertices(boxB)
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
    intersection_points = sort_points_clockwise(intersection_points)

    return intersection_points

def intersection_area_diy(intersection_points):
    """
    Compute the Area using Triangle Formula byMauren Abreu de Souza
    Args: list of intersection points with shape (1,p)
    Returns: intersection area with shape (1,1)
    """
    len_points = len(intersection_points)
    area = 0.0
    for i in range(len_points):
        j = (i + 1) % len_points
        area += intersection_points[i][0] * intersection_points[j][1]
        area -= intersection_points[j][0] * intersection_points[i][1]

    area = abs(area) / 2.0

    return area

## ==============================================================================================================
## ============================================ DEFINE THE JACOBIAN MATRICES ====================================
## ==============================================================================================================

# J_alpha, Jacobian Matrix that mapping from intersection points/area to image space
def construct_J_alpha(intersection_points):
    len_points = len(intersection_points)
    J_alpha = []
    for i in range(len_points):
        j = (i + 1) % len_points
        k = (i + len_points) % len_points
        J_alpha.append(intersection_points[j][1] - intersection_points[k][1])
        J_alpha.append(intersection_points[k][0] - intersection_points[j][0])

    return J_alpha

# J_o, Jacobian Matrix which maps xywhr format space into the image space (image feature vector with xyxyxy format)
J_o = torch.tensor()

## ==============================================================================================================
## ==================================== FOR TESTING PURPOSES =================================================
## ==============================================================================================================

boxA = np.array([0, 0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8])
boxB = np.array([0, 0.5, 0.9, 0.9, 0.9, 0.9, 1.0, 0.5, 1.0])
boxC = np.array([0, 0.2, 0.1, 0.4, 0.1, 0.4, 0.9, 0.2, 0.9])
boxD = np.array([0, 0.6, 0.1, 0.9, 0.4, 0.7, 0.8, 0.4, 0.5])
boxE = np.array([0, 0.2, 0.1, 0.4, 0.3, 0.3, 0.4, 0.1, 0.2])
print(cxyxyxyxy2xywhr(boxA))
print(cxyxyxyxy2xywhr(boxB))
print(cxyxyxyxy2xywhr(boxC))
print(cxyxyxyxy2xywhr(boxD))
print(cxyxyxyxy2xywhr(boxE))
print("------------------------")
print("DIY Area of boxA and boxE", intersection_area_diy(intersection_points_diy(boxA, boxE)))
print("Shapely Area of boxA and boxE", intersection_area_shapely(boxA, boxE))
print("------------------------")
interp = intersection_points_diy(boxA, boxE)
J_alpha = construct_J_alpha(interp)
print(interp)
print(J_alpha)

print("image feature vector of box A", cxyxyxyxy2xyxyxy(boxA))


