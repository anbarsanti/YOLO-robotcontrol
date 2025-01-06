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
hbbA_xywh = np.array([1, 0.7, 0.3, 0.4, 0.4]) #xywh format
hbbB_xywh = np.array([1, 0.8, 0.2, 0.2, 0.2]) #xywh format
hbbC_xywh = np.array([1, 0.35, 0.6, 0.5, 0.4]) #xywh format
hbbD_xyxy = np.array([0, 0.2, 0.2, 0.8, 0.8])
hbbE_xyxy = np.array([0, 0.3, 0.1, 0.5, 0.9])
hbbF_xyxy = np.array([0, 0.6, 0.4, 0.9, 0.6])

print("area HBB A and B", intersection_area_HBB_xywh(hbbA_xywh, hbbB_xywh))
print("area HBB A and C", intersection_area_HBB_xywh(hbbA_xywh, hbbC_xywh))
print("area HBB B and C", intersection_area_HBB_xywh(hbbB_xywh, hbbC_xywh))

print("area HBB D and E", intersection_area_HBB_xyxy(hbbD_xyxy, hbbE_xyxy))
print("area HBB D and F", intersection_area_HBB_xyxy(hbbD_xyxy, hbbF_xyxy))
print("area HBB E and F", intersection_area_HBB_xyxy(hbbE_xyxy, hbbF_xyxy))



# print("check vertices hbbA_xywh", convert_HBB_xywh_to_vertices(hbbA_xywh))

# reaching_box = [0.0, 0.59080827236, 0.2633193731079834, 0.844, 0.617902934551239] # Toys
# desired_box = [1.0, 0.36258264, 0.5401297211647, 0.5640562176704407, 0.8815667033195496] # Box
# hbbF = [1, 0,0,0,0]
# print(is_HBB_intersect(reaching_box, desired_box))
# print("Area between reaching box and desired box", intersection_area_HBB(reaching_box, desired_box))

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