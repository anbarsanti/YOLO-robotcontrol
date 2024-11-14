"""
﷽
author: @anbarsanti
"""

import numpy as np
import math
from sympy import *

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
