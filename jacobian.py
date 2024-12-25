"""
﷽
by @anbarsanti
"""

import numpy as np
import math
import torch
from sympy import *

## ==============================================================================================================
## ============================================ DEFINE THE JACOBIAN MATRICES ====================================
## ==============================================================================================================

# J_alpha, Jacobian Matrix that mapping from intersection points/area to image space
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
   return np.array(image_jacobian)

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
   return np.array(image_jacobian)

def J_I_6x6 (xc, yc, x1, y1, x2, y2, depth):
   """
   Return the Jacobian matrix that maps 3 points in image space to linear velocity and angular velocity to cartesian space.
   Args:
       3 points and depth
   Returns:
       J_I (6x6 matrix)
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
   return np.array(image_jacobian)

def J_I_8x6 (x1, y1, x2, y2, x3, y3, x4, y4, depth):
   """
   Return the Jacobian matrix that maps 4 points in image space to linear velocity and angular velocity to cartesian space.
   Args:
       4 points and depth
   Returns:
       J_I (8x6 matrix)
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
   return np.array(image_jacobian)

def J_a(intersection_points, depth):
   """
   Construct the Jacobian matrix that maps intersection points in image space to linear velocity and angular velocity to cartesian space.
   Args:
       a number of intersection points
   Returns:
       J_I (8x3 matrix)
   """
   len_points = len(intersection_points)
   p = intersection_points
   jacobian = []
   f_x = 618.072
   f_y = 618.201
   for i in range(len_points):
      j = [f_x/depth, 0, -p[i][0]/depth, -p[i][0]*p[i][1]/f_x, (f_x*f_x + p[i][0]*p[i][0])/f_x, -p[i][1]]
      k = [0, f_y/depth, -p[i][1]/depth, -(f_y*f_y + p[i][1]*p[i][1])/f_y, p[i][0]*p[i][1]/f_y, p[i][0]]
      jacobian.extend([[j],[k]])
   
   # Compute the determinant to check singularity
   J_a = np.array(jacobian)
   # determinant = np.linalg.det(J_a)
   
   return J_a

def J_o(p):
   """
   Construct the Jacobian matrix that maps yolo points to image feature points (3 points)
   Args:
       image feature points [x1, y1, x2, y2, x3, y3]
       p_1 and p_2 represent the midpoint on the adjacent two sides of the rotating bounding box
       p_3 denotes the center of rotation bounding box
   Returns:
      The jacobian matrix J_o (5x4 matrix)
   """
   # Precompute common terms
   jacobian = np.zeros((5, 6))
   x1 = p[0]; y1 = p[1]; x2 = p[2]; y2 = p[3]; x3 = p[4]; y3 = p[5]
   delta_x1 = x3-x1; delta_y1 = y1-y3; delta_y2 = y3-y2; delta_x2 = x3-x2; theta = np.arctan2(delta_y1, delta_x1)
   denom1 = delta_x1**2 + delta_y1**2; sqrt_denom1 = np.sqrt(denom1)
   denom2 = delta_x2**2 + delta_y2**2; sqrt_denom2 = np.sqrt(denom2)
   
   # Ensure inputs are not zero to avoid division by zero
   assert denom1 > 0, "Invalid input: delta_x1^2 + delta_y1^2 must be > 0"
   assert denom2 > 0, "Invalid input: delta_x2^2 + delta_y2^2 must be > 0"
   
   # Define the jacobian
   jacobian [0,:] = [0, 0, 0, 0, 1, 0]
   jacobian [1,:] = [0, 0, 0, 0, 0, 1]
   jacobian [2,:] = [-2*delta_x1/sqrt_denom1, 2*delta_y1/sqrt_denom1, 0, 0, 2*delta_x1/sqrt_denom1, -2*delta_y1/sqrt_denom1]
   jacobian [3,:] = [0, 0, -2*delta_x2/sqrt_denom2, -2*delta_y2/sqrt_denom2, 2*delta_x2/sqrt_denom2, 2*delta_y2/sqrt_denom2]
   jacobian [4,:] = [delta_y1/denom1, delta_x1/denom1, 0, 0, -delta_y1/denom1, -delta_x1/denom1]
   
   # Compute the determinant to check singularity
   # determinant = np.linalg.det(jacobian)
   
   return jacobian

def numerical_J_o(p, epsilon=1e-8):
   """
   Use numerical approximation to verify the analytical Jacobian matrix J_o
   Args:
      image feature points [x1, y1, x2, y2, x3, y3]
      p_1 and p_2 represent the midpoint on the adjacent two sides of the rotating bounding box
      p_3 denotes the center of rotation bounding box
   Returns: assertion results
   """
   # Define the Sigma function
   def sigma(p):
      # Precompute common terms
      x1 = p[0]; y1 = p[1]; x2 = p[2]; y2 = p[3]; x3 = p[4]; y3 = p[5]
      delta_x1 = x3 - x1; delta_y1 = y1 - y3; delta_y2 = y3 - y2; delta_x2 = x3 - x2; theta = np.arctan2(delta_y1, delta_x1)
      denom1 = delta_x1 ** 2 + delta_y1 ** 2; sqrt_denom1 = np.sqrt(denom1)
      denom2 = delta_x2 ** 2 + delta_y2 ** 2; sqrt_denom2 = np.sqrt(denom2)
      return [x3, y3, 2*sqrt_denom1, 2*sqrt_denom2, theta]
   
   # Build the Jacobian matrix using numerical approach
   n = len(p)
   m = len(sigma(p))
   jacobian = np.zeros((m, n))
   
   for i in range(n):
      p_plus = p.copy()
      p_plus[i] += epsilon
      p_minus = p.copy()
      p_minus[i] -= epsilon
      jacobian[:,i] = np.subtract(sigma(p_plus),sigma(p_minus)) / (2*epsilon)
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
	   q: List or array of joint angles [q1, q2, q3, q4, q5, q6]
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
	   q: List or array of joint angles [q1, q2, q3, q4, q5, q6]
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
	   q: List or array of joint angles [q1, q2, q3, q4, q5, q6]
   Returns:
      Jacobian matrix J_r (6x6 matrix), including linear and angular velocity parts
   """
   # Precompute & Predefine some terms
   pi = 3.1415926535
   jacobian = np.zeros((6, 6))
   d1 = 0.0892 # 0.1625
   d4 = 0.1093 # 0.1333
   d5 = 0.09475 # 0.0997
   d6 = 0.0825 # 0.0996
   a2 = 0.425
   a3 = 0.392
   q1 = q[0]; q2 = q[1]; q3 = q[2]; q4 = q[3]; q5 = q[4]; q6 = q[5]
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
   px = (r13*d6) + (c1*((s234*d5) + (c23*a3) + (c2*a2))) + (s1*d4)
   py = (r23*d6) + (s1*((s234*d5) + (c23*a3) + (c2*a2))) - (c1*d4)
   pz = (r33*d6) - (c234*d5) + (s23*a3) + (s2*a2) + d1
   
   # Define the Jacobian Matrix
   jacobian[0,0]= -py; jacobian[0,1] = -c1*(pz-d1);  jacobian[0,2]= c1*(s234*s5*d6+(c234*d5)-(s23*a3));    jacobian[0,3] = c1*((s234*s5*d6)+(c234*d5)); jacobian[0,4]= -d6*((s1*s5)+(c1*c234*c5)); jacobian[0,5]= 0
   jacobian[1,0]= px;  jacobian[1,1] = -s1*(pz-d1);   jacobian[1,2]= s1*((s234*s5*d6)+(c234*d5)-(s23*a3)); jacobian[1,3] = s1*((s234*s5*d6)+(c234*d5)); jacobian[1,4] = d6*((c1*s5)-(c234*c5*s1)); jacobian[1,5]= 0
   jacobian[2,0]= 0;   jacobian[2,1] = (s1*py)+(c1*px); jacobian[2,2]= -(c234*s5*d6)+(s234*d5)+(c23*a3);   jacobian[2,3] = -(c234*s5*d6)+(s234*d5);     jacobian[2,4] = -c5*s234*d6;               jacobian[2,5]= 0
   
   jacobian[3,0] = 0; jacobian[3,1] = s1;  jacobian[3,2] = s1;  jacobian[3,3] = s1;  jacobian[3,4] = c1*s234; jacobian[3,5] = r13
   jacobian[4,0] = 0; jacobian[4,1] = -c1; jacobian[4,2] = -c1; jacobian[4,3] = -c1; jacobian[4,4] = s1*s234; jacobian[4,5] = r23
   jacobian[5,0] = 1; jacobian[5,1] = 0;   jacobian[5,2] = 0;   jacobian[5,3] = 0;   jacobian[5,4] = -c234;   jacobian[5,5] = r33
   
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

