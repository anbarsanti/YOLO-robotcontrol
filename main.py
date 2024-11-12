# ﷽
import numpy as np
import math
import PolygonCollision # Separating Axis Theorem from https://pypi.org/project/PolygonCollision/ --> does not work
from ultralytics.data.converter import convert_dota_to_yolo_obb


# YOLO OBB format is class_index x1 y1 x2 y2 x3 y3 x4 y4
# x1, y1: Coordinates of the first corner point
# x2, y2: Coordinates of the second corner point
# x3, y3: Coordinates of the third corner point
#  x4, y4: Coordinates of the fourth corner point
# All coordination are normalized to values between 0 and 1 relative to image dimensions (allows the annotation to be resolution dependent)
# YOLO v11 OBB format is already in polygon format

boxA = np.array([0, 0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])
boxB = np.array([0, 0.5, 0.1, 0.9, 0.5, 0.5, 0.9, 0.1, 0.5])
boxC = np.array([0, 0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.7, 0.3])
boxD = np.array([0, 0.3, 0.3, 0.4, 0.3, 0.4, 0.4, 0.3, 0.4])
boxE = np.array([0, 0.7, 0.7, 0.8, 0.7, 0.8, 0.8, 0.7, 0.8])



def convert_to_vertices(boxA):
    # Change the format of array, from [0,1,2,3,4,5,6,7,8] to [[1,2], [3,4], [5,6], [7,8]]
    vertices = [[boxA[i], boxA[i + 1]] for i in range(1, len(boxA), 2)]
    return vertices

def edges_of(vertices):
    # Return the vectors for the edges of the polygon
    edges = []
    N = len(vertices)
    for i in range(N):
        edge = np.subtract(vertices[(i+1) % N],vertices[i]).tolist()
        edges.append(edge)
    return edges

def orthogonal(v):
    # Return a 90 degree clockwise rotation of the vector v
    return np.array([-v[1], v[0]])

def project(vertices, axis):
    # project vertices [[1,2], [3,4], [5,6], [7,8]] onto axis [x,y]
    dots = [np.dot(vertex, axis) for vertex in vertices]/np.linalg.norm(axis)
    return [min(dots), max(dots)]

def project_overlap(projection1, projection2):
    # Check if projections1 [min1, max1] overlap with projection2[min2, max2]
    # Overlap occur if and only if min1 < max2 and min2 < max1
    return (projection1[0] < projection2[1]) and (projection2[0] < projection2[1])

def is_intersect(boxA, boxB):
    # Determining if two OBBs are intersecting using Separating Axis Theorem
    edges = edges_of(convert_to_vertices(boxA)) + edges_of(convert_to_vertices(boxB))
    axes = [orthogonal(edge) for edge in edges]
    for axis in axes:
        projectionA = project(convert_to_vertices(boxA), axis)
        projectionB = project(convert_to_vertices(boxB), axis)
        if not project_overlap(projectionA, projectionB):
            return False # box A and box B not overlapping, separating axis found
    return True # box A and box B is overlapping, no separating axis found


# # --------- Checking Purpose
# print("vertices of A:", convert_to_vertices(boxA))
# print("edges(vector) of A:", edges_of(convert_to_vertices(boxA)))
# print("first edge of A:", edges_of(convert_to_vertices(boxA))[0])
# print("orthogonal of that edge:", orthogonal(edges_of(convert_to_vertices(boxA))[0]))
# print("projection box A onto a first edge of A", project(convert_to_vertices(boxA), edges_of(convert_to_vertices(boxA))[0]))
# print("projection box B onto a first edge of A", project(convert_to_vertices(boxB), edges_of(convert_to_vertices(boxA))[0]))
# print(project_overlap(convert_to_vertices(boxA), convert_to_vertices(boxB)))
# print(is_intersect(boxA, boxB))
# print(is_intersect(boxA, boxC))
# print(is_intersect(boxB, boxC))
# print(is_intersect(boxD, boxE))
