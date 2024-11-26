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
    len_points = len(intersection_points)
    J_alpha = []
    for i in range(len_points):
        j = (i + 1) % len_points
        k = (i + len_points) % len_points
        J_alpha.append(intersection_points[j][1] - intersection_points[k][1])
        J_alpha.append(intersection_points[k][0] - intersection_points[j][0])

    return J_alpha

# J_o, Jacobian Matrix which maps xywhr format space into the image space (image feature vector with xyxyxy format)
# J_o = torch.tensor()


init_printing()

x = MatrixSymbol('x', 4, 1)
f = Matrix([[x[0]*x[1]],
           [sin(x[0])],
           [cos(x[2])],
           [x[2]*E**(x[3])]])

JacobianMatrixSymbolic=f.jacobian(x)

JacobianFunction = lambdify(x, JacobianMatrixSymbolic)

xTest = np.array([[1],
                  [1],
                  [1],
                  [1]])
