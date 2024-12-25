"""
﷽
by @anbarsanti
"""

import numpy as np
import math
import torch
from r2r_functions import *


## ==============================================================================================================
## ==================================== TESTING PURPOSES =================================================
## ==============================================================================================================

# ------------------------------ HBB ---------------------------------
# print("----------------------HBB Test--------------------------")
# hbbA = np.array([1, 0.7, 0.3, 0.4, 0.4])
# hbbC = np.array([1, 0.8, 0.8, 0.2, 0.2])
# hbbD = np.array([1, 0.7, 0.55, 0.2, 0.3])
# print("HBB vertices of hbbA", convert_HBB_to_vertices(hbbA))
# print("HBB vertices of hbbA", convert_HBB_to_vertices(hbbC))
# print("HBB vertices of hbbA", convert_HBB_to_vertices(hbbD))
# print(intersection_area_HBB(hbbA, hbbC))
# print(intersection_area_HBB(hbbA, hbbD))
# print(intersection_area_HBB(hbbC, hbbD))

# ------------------------------ OBB ---------------------------------
print("----------------------OBB Test--------------------------")
obbA = np.array([0, 0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8])
obbB = np.array([0, 0.5, 0.9, 0.9, 0.9, 0.9, 1.0, 0.5, 1.0])
obbC = np.array([0, 0.2, 0.1, 0.4, 0.1, 0.4, 0.9, 0.2, 0.9])
obbD = np.array([0, 0.6, 0.1, 0.9, 0.4, 0.7, 0.8, 0.4, 0.5])
obbE = np.array([0, 0.2, 0.1, 0.4, 0.3, 0.3, 0.4, 0.1, 0.2])
print("DIY Area of boxA and boxD", intersection_area_OBB_diy(obbA, obbD))
# print("Shapely Area of boxA and boxD", intersection_area_OBB_shapely(obbA, obbD))
interp = intersection_points_OBB_diy(obbA, obbD)
P_R, P_R_mat = r2r_controller(obbD, obbE, OBB=True)
print("P_R", P_R)
print("P_R_mat", P_R_mat)

# # ------------------------------ Jacobian Test ---------------------------------
# print("----------------------Jacobian Test--------------------------")
# p = np.array([0.5, 0.9, 0.2, 0.1, 0.4, 0.7])
# print("Analytical jacobian J_o:", J_o(p))
# print("Numerical jacobian J_o:", numerical_J_o(p))
