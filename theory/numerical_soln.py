import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.stats import norm
import pde
from pde import CartesianGrid, solve_laplace_equation
import sympy
from sympy import stats

z=1
sigma=1
def u0(x,z=1,sigma=1):
    return z*norm.pdf(x,loc=0,scale=sigma)

# bc_x_left = {"value": 1} # time
# bc_x_right = {"derivative": "0"} # time
# bc_x = [bc_x_left, bc_x_right] # space
# bc_y = "periodic" # space

grid = pde.CartesianGrid([[-10,10]],[20],periodic=[True]) #t,x dimensions (periodic BC in x only)
f = stats.Normal("z",0,sigma)
x = sympy.Symbol("x")
state = pde.ScalarField.from_expression(grid, stats.density(f)('x'))
# field = pde.ScalarField(grid)
eq = pde.PDE({"u": "laplace(u)-u+u*u"},bc=["periodic"])
result = eq.solve(state,t_range=10,dt=1e-2)

# bc_x_left = {"value": }
# bc_x_right = {"value": "sin(y / 2)"}
# bc_x = [bc_x_left, bc_x_right]
# bc_y = "periodic"
# eq = DiffusionPDE(bc=[bc_x, bc_y])
#
#
# bc = [{"value": "1/(2*sqrt(2*pi))*E**(-x**2/2)"},"periodic"] #z=1,sigma=1
# field.laplace(bc=bc)
#

#
# result = eq.solve(field,t_range=10,dt=1e-2)
result.plot()
#
#
#
