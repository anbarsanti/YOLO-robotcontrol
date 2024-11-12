# ﷽
import numpy as np
import PolygonCollision # Separating Axis Theorem from https://pypi.org/project/PolygonCollision/ --> does not work
from ultralytics.data.converter import convert_dota_to_yolo_obb


# YOLO OBB format is class_index x1 y1 x2 y2 x3 y3 x4 y4
# x1, y1: Coordinates of the first corner point
# x2, y2: Coordinates of the second corner point
# x3, y3: Coordinates of the third corner point
#  x4, y4: Coordinates of the fourth corner point
# All coordination are normalized to values between 0 and 1 relative to image dimensions (allows the annotation to be resolution dependent)
# YOLO v11 OBB format is already in polygon format

boxA = np.array([0, 0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.8, 0.2])
boxB = np.array([0, 0.5, 0.1, 0.9, 0.5, 0.5, 0.9, 0.1, 0.5])
boxC = np.array([0, 0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.7, 0.3])


def convert_vertices(boxA):
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
    # Return a 90 degree counter clockwise rotation of the vector v
    return np.array([-v[1], v[0]])




# Check function by function
print("vertices:", convert_vertices(boxA))
print("edges(vector):", edges_of(convert_vertices(boxA)))
print("an edge:", edges_of(convert_vertices(boxA))[1])
print("orthogonal of the edge:", orthogonal(edges_of(convert_vertices(boxA))[1]))

# Check using PolygonCollision

polygonA = PolygonCollision.shape.Shape(vertices=convert_vertices(boxA),fill=False)
polygonB = PolygonCollision.shape.Shape(vertices=convert_vertices(boxB),fill=False)
polygonC = PolygonCollision.shape.Shape(vertices=convert_vertices(boxC),fill=False)

print(polygonA.collide(polygonB))
print(polygonA.collide(polygonC))
print(polygonB.collide(polygonC))


