from fipy import Variable, FaceVariable, CellVariable, Grid1D, ExplicitDiffusionTerm, TransientTerm, DiffusionTerm, Viewer
from fipy.tools import numerix

nx = 50
dx = 1.
mesh = Grid1D(nx = nx, dx = dx)
phi = CellVariable(name="solution variable",
                   mesh=mesh,
                   value=0)

D0=1.
valueLeft = 1.
valueRight = 0.

eq = DiffusionTerm(coeff=D0 * (1 - phi[0]))
phi[0].setValue(valueRight)
res = 1e+10
while res > 1e-6:
    res = eq.sweep(var=phi[0],
                   dt=timeStepDuration)

print(phi[0].allclose(phiAnalytical, atol = 1e-1))
# Expect:
# 1
#
if __name__ == '__main__':
    viewer.plot()
    input("Implicit variable diffusivity - steady-state. \
Press <return> to proceed...")

