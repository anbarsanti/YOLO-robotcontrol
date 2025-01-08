"""
﷽
by @anbarsanti
"""
import sys

from YOLOv11.r2r_functions import intersection_points_HBB_xyxy

sys.path.append('../RTDE_Python_Client_Library')
from r2r_functions import *
import numpy as np
## ====================================  HBB TEST ====================================
## print("==================================== HBB Test ==================================== ")
# hbbA_xywh = np.array([1, 0.7, 0.3, 0.4, 0.4]) #xywh format
# hbbB_xywh = np.array([1, 0.8, 0.2, 0.2, 0.2]) #xywh format
# hbbC_xywh = np.array([1, 0.35, 0.6, 0.5, 0.4]) #xywh format
hbbD_xyxy = np.array([0, 0.2, 0.2, 0.8, 0.8])
hbbE_xyxy = np.array([0, 0.3, 0.1, 0.5, 0.9])
hbbF_xyxy = np.array([0, 0.6, 0.4, 0.9, 0.6])
# #
# # print("area HBB A and B", intersection_area_HBB_xywh(hbbA_xywh, hbbB_xywh))
# # print("area HBB A and C", intersection_area_HBB_xywh(hbbA_xywh, hbbC_xywh))
# # print("area HBB B and C", intersection_area_HBB_xywh(hbbB_xywh, hbbC_xywh))
# #
# # print("area HBB D and E", intersection_area_HBB_xyxy(hbbD_xyxy, hbbE_xyxy))
# # print("area HBB D and F", intersection_area_HBB_xyxy(hbbD_xyxy, hbbF_xyxy))
# # print("area HBB E and F", intersection_area_HBB_xyxy(hbbE_xyxy, hbbF_xyxy))
#
# print("hbbD in xywhr format", cxyxy2xywhr(hbbD_xyxy))
# print("hbbE in xywhr format", cxyxy2xywhr(hbbE_xyxy))
# print("hbbF in xywhr format", cxyxy2xywhr(hbbF_xyxy))
#
q = [    0.39781,    -0.60547,   0.0021877,   0.0076568,   0.0032815,  -0.0010938]
# q_reshape = np.array(q).reshape((-1,1))
# print("q_reshape", q_reshape)
transposed = np.array(q).reshape(-1,1)
print(transposed)




### ====================================  OBB ====================================
# print("==================================== OBB Test ==================================== ")
obbA = np.array([0, 0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8])
obbB = np.array([0, 0.5, 0.9, 0.9, 0.9, 0.9, 1.0, 0.5, 1.0])
obbC = np.array([0, 0.2, 0.1, 0.4, 0.1, 0.4, 0.9, 0.2, 0.9])
obbD = np.array([0, 0.6, 0.1, 0.9, 0.4, 0.7, 0.8, 0.4, 0.5])
obbE = np.array([0, 0.2, 0.1, 0.4, 0.3, 0.3, 0.4, 0.1, 0.2])
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
# sorted_p1 = sort_points_clockwise(p1)
# print("sorted_p1", sorted_p1)
# p2 = intersection_points_OBB_diy(obbA, obbD)
# print("interp A and D:", p2)
# p3 = intersection_points_OBB_diy(obbA, obbE)
# print("interp A and E:", p3)
# p4 = intersection_points_OBB_diy(obbC, obbD)
# print("interp C and D:", p4)
# p5 = intersection_points_OBB_diy(obbC, obbE)
# print("interp C and E:", p5)

# # # ------------------------------ Jacobian Test ---------------------------------
# # print("----------------------Jacobian Test--------------------------")
q = np.array([0.23, 0.91, 0.22, 0.12, 0.42, 0.74]).reshape((-1,1))
p_r_hbb = cxyxy2xyxyxy(hbbD_xyxy)
p_r_obb = cxyxyxyxy2xyxyxy(obbB)

# print("q_dot", q_dot)
# print("epsilon_A", epsilon_A)
# print("epsilon_S", epsilon_S)
# print("J_o_I_r", J_o_I_r)
# print("J_alpha_a_r", J_alpha_a_r)

# ## ===================== J_o_I_r Testing =====================
# print("J_o(p_r_box)", J_o(p_r_obb))
# print("J_o(p_r_box).shape", J_o(p_r_obb).shape)
# print("J_I(p_r_box)", J_I(p_r_obb))
# print("J_I(p_r_box).shape", J_I(p_r_obb).shape)
# print("J_r(q)", J_r(q))
# print("J_r(q).shape", J_r(q).shape)
J_o_I_r = (J_o(p_r_obb)) @ (J_I(p_r_obb)) @ (J_r(q))
# print("J_o_I_r", J_o_I_r)
print("J_o_I_r.shape", J_o_I_r.shape)
J_o_I_r_pinv = np.linalg.pinv(J_o_I_r)
print("J_o_I_r_pinv", J_o_I_r_pinv)
print("J_o_I_r_pinv.shape", J_o_I_r_pinv.shape)
# J_o_I_r_transpose = J_o_I_r.T
# print("J_o_I_r_transpose", J_o_I_r_transpose)
# print("J_o_I_r_transpose.shape", J_o_I_r_transpose.shape)

## ===================== J_olpha_a_r Testing =====================
p1 = intersection_points_HBB_xyxy(hbbD_xyxy, hbbF_xyxy)
# # print("interp D and E:", p1)
# # print("J_alpha(p1)", J_alpha(p1))
# # print("J_alpha(p1).shape", J_alpha(p1).shape)
# # print("J_a(p1)", J_a(p1))
# # print("J_a(p1).shape", J_a(p1).shape)
# # print("J_r(q)", J_r(q))
# # print("J_r(q).shape", J_r(q).shape)
J_alpha_a_r = ((J_alpha(p1)) @ (J_a(p1)) @ (J_r(q))).reshape(1,6)
print("J_alpha_a_r.shape", J_alpha_a_r.shape)
J_alpha_a_r_pinv = np.linalg.pinv(J_alpha_a_r)
print("J_alpha_a_r_pinv", J_alpha_a_r_pinv)
print("J_alpha_a_r_pinv.shape", J_alpha_a_r_pinv.shape)

jacobian = np.concatenate((J_alpha_a_r_pinv, J_o_I_r_pinv), axis=1)
print("jacobian", jacobian)
print("jacobian.shape", jacobian.shape)


q_dot, epsilon= r2r_control(obbD, obbE, q, OBB=True)
print("epsilon", epsilon)
print("epsilon.shape", epsilon.shape)
print("q_dot", q_dot)
print("q_dot.shape", q_dot.shape)
