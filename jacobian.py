"""
﷽
author: @anbarsanti
"""

import numpy as np
import math
import torch
from sympy import *

## ==============================================================================================================
## ============================================ DEFINE THE JACOBIAN MATRICES ====================================
## ==============================================================================================================

# J_alpha, Jacobian Matrix that mapping from intersection points/area to image space
def construct_J_alpha(intersection_points):
    """
    Return J_alpha, the Jacobian matrix that maps the area from intersection points to image space.
    Args:
        the vertices of the polygon
    Returns:
        J_alpha
    """
    len_points = len(intersection_points)
    J_alpha = []
    for i in range(len_points):
        j = (i + 1) % len_points
        k = (i + len_points) % len_points
        J_alpha.append(intersection_points[j][1] - intersection_points[k][1])
        J_alpha.append(intersection_points[k][0] - intersection_points[j][0])

    return J_alpha

def J_I_8x3 (x1, y1, x2, y2, x3, y3, x4, y4, depth):
   """
   Return the Jacobian matrix that maps 4 points in image space to linear velocity cartesian space.
   Args:
       4 points and depth
   Returns:
       J_I (8x3 matrix)
   """
   f_x = 618.072
   f_y = 618.201
   image_jacobian = [
      [-f_x / depth, 0, x1 / depth],
      [0, -f_y / depth, y1 / depth],
      [-f_x / depth, 0, x2 / depth],
      [0, -f_y / depth, y2 / depth],
      [-f_x / depth, 0, x3 / depth],
      [0, -f_y / depth, y3 / depth],
      [-f_x / depth, 0, x4 / depth],
      [0, -f_y / depth, y4 / depth]
   ]
   return image_jacobian

def J_I_6x3 (xc, yc, x1, y1, x2, y2, depth):
   """
   Return the Jacobian matrix that maps 3 points in image space to linear velocity cartesian space.
   Args:
       3 points and depth
   Returns:
       J_I (6x3 matrix)
   """
   f_x = 618.072
   f_y = 618.201
   image_jacobian = [
      [-f_x / depth, 0, xc / depth],
      [0, -f_y / depth, yc / depth],
      [-f_x / depth, 0, x1 / depth],
      [0, -f_y / depth, y1 / depth],
      [-f_x / depth, 0, x2 / depth],
      [0, -f_y / depth, y2 / depth],
   ]
   return image_jacobian

def J_I_6x6 (xc, yc, x1, y1, x2, y2, depth):
   """
   Return the Jacobian matrix that maps 3 points in image space to linear velocity and angular velocity to cartesian space.
   Args:
       3 points and depth
   Returns:
       J_I (8x3 matrix)
   """
   f_x = 618.072
   f_y = 618.201
   # print('depth',depth)
   
   image_jacobian = [
      [-f_x / depth, 0, xc / depth, xc * yc / f_x, -(f_x * f_x + xc * xc) / f_x, yc],
      [0, -f_y / depth, yc / depth, (f_y * f_y + yc * yc) / f_y, -xc * yc / f_y, -xc],
      [-f_x / depth, 0, x1 / depth, x1 * y1 / f_x, -(f_x * f_x + x1 * x1) / f_x, y1],
      [0, -f_y / depth, y1 / depth, (f_y * f_y + y1 * y1) / f_y, -x1 * y1 / f_y, -x1],
      [-f_x / depth, 0, x2 / depth, x2 * y2 / f_x, -(f_x * f_x + x2 * x2) / f_x, y2],
      [0, -f_y / depth, y2 / depth, (f_y * f_y + y2 * y2) / f_y, -x2 * y2 / f_y, -x2],
   ]
   return image_jacobian

def J_I_8x6 (x1, y1, x2, y2, x3, y3, x4, y4, depth):
   """
   Return the Jacobian matrix that maps 4 points in image space to linear velocity and angular velocity to cartesian space.
   Args:
       3 points and depth
   Returns:
       J_I (8x3 matrix)
   """
   f_x = 618.072
   f_y = 618.201
   image_jacobian = [
      [f_x / depth, 0, - x1 / depth, -x1 * y1 / f_x, (f_x * f_x + x1 * x1) / f_x, -y1],
      [0, f_y / depth, - y1 / depth, -(f_y * f_y + y1 * y1) / f_y, x1 * y1 / f_y, x1],
      [f_x / depth, 0, - x2 / depth, -x2 * y2 / f_x, (f_x * f_x + x2 * x2) / f_x, -y2],
      [0, f_y / depth, - y2 / depth, -(f_y * f_y + y2 * y2) / f_y, x2 * y2 / f_y, x2],
      [f_x / depth, 0, - x3 / depth, -x3 * y3 / f_x, (f_x * f_x + x3 * x3) / f_x, -y3],
      [0, f_y / depth, - y3 / depth, -(f_y * f_y + y3 * y3) / f_y, x3 * y3 / f_y, x3],
      [f_x / depth, 0, - x4 / depth, -x4 * y4 / f_x, (f_x * f_x + x4 * x4) / f_x, -y4],
      [0, f_y / depth, - y4 / depth, -(f_y * f_y + y4 * y4) / f_y, x4 * y4 / f_y, x4]
   ]
   return image_jacobian

def construct_J_a(intersection_points):
   len_points = len(intersection_points)
   J_a = []
   for i in range(len_points):
      j = (i + 1) % len_points
      k = (i + len_points) % len_points
      J_a.append(intersection_points[j][0] - intersection_points[k][0])
      J_a.append(intersection_points[k][1] - intersection_points[j][1])
   return J_a