"""
﷽
by @anbarsanti
"""

import numpy as np
import math
import torch
from r2r_functions import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

## ==============================================================================================================
## ==================================== TESTING PURPOSES =================================================
## ==============================================================================================================

# # ------------------------------ HBB ---------------------------------
# print("----------------------HBB Test--------------------------")
# hbbA = np.array([1, 0.7, 0.3, 0.4, 0.4])
# hbbB = np.array([1, 0.8, 0.2, 0.2, 0.2])
# hbbC = np.array([1, 0.35, 0.6, 0.5, 0.4])
reaching_box = [0.0, 0.5717689394950867, 0.35506153106689453, 0.8555841445922852, 0.7349398136138916]
desired_box = [1.0, 0.14817076921463013, 0.5333613753318787, 0.391799658536911, 0.8300615549087524]
hbbF = [1, 0,0,0,0]
print(is_HBB_intersect(reaching_box, desired_box))
print("Area between reaching box and desired box", intersection_area_HBB(reaching_box, desired_box))

# print("Area A vs D", intersection_area_HBB(hbbA, hbbC))
# print("Area B vs D", intersection_area_HBB(hbbB, hbbC))
# print("image feature points of hbbA:", cxywh2xyxyxy(hbbA))

# # # ------------------------------ OBB ---------------------------------
# print("----------------------OBB Test--------------------------")
# obbA = np.array([0, 0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8])
# obbB = np.array([0, 0.5, 0.9, 0.9, 0.9, 0.9, 1.0, 0.5, 1.0])
# obbC = np.array([0, 0.2, 0.1, 0.4, 0.1, 0.4, 0.9, 0.2, 0.9])
# obbD = np.array([0, 0.6, 0.1, 0.9, 0.4, 0.7, 0.8, 0.4, 0.5])
# obbE = np.array([0, 0.2, 0.1, 0.4, 0.3, 0.3, 0.4, 0.1, 0.2])
# print("Area of A and C", intersection_area_OBB_diy(obbA, obbC))
# print("Area of A and C shapely", intersection_area_OBB_shapely(obbA, obbC))
# print("Area of A and D", intersection_area_OBB_diy(obbA, obbD))
# print("Area of A and D shapely", intersection_area_OBB_shapely(obbA, obbD))
# print("Area of A and E", intersection_area_OBB_diy(obbA, obbE))
# print("Area of A and E shapely", intersection_area_OBB_shapely(obbA, obbE))
# print("Area of C and D", intersection_area_OBB_diy(obbC, obbD))
# print("Area of C and D shapely", intersection_area_OBB_shapely(obbC, obbD))
# print("Area of C and E", intersection_area_OBB_diy(obbC, obbE))
# print("Area of C and E shapely", intersection_area_OBB_shapely(obbC, obbE))
# p1 = intersection_points_OBB_diy(obbA, obbC)
# print("interp A and C:", p1)
# p2 = intersection_points_OBB_diy(obbA, obbD)
# print("interp A and D:", p2)
# p3 = intersection_points_OBB_diy(obbA, obbE)
# print("interp A and E:", p3)
# p4 = intersection_points_OBB_diy(obbC, obbD)
# print("interp C and D:", p4)
# p5 = intersection_points_OBB_diy(obbC, obbE)
# print("interp C and E:", p5)

#
#
# # # ------------------------------ Jacobian Test ---------------------------------
# # print("----------------------Jacobian Test--------------------------")
# q = np.array([0.23, 0.91, 0.22, 0.12, 0.42, 0.74]).reshape((-1,1))
# q_dot = r2r_control(obbD, obbE, q, OBB=True)

## ==============================================================================================================
## TRACKING
## ==============================================================================================================