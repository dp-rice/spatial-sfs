from fipy import *
from fipy import meshes
from scipy.stats import norm
import numpy as np

def u0(x, z=1., lc=50,sigma=1.):
    """
    Function to return initial condition value
    :param x: space variable from mesh
    :param z: dummy variable (given)
    :param lc: mean of normal distribution (given)
    :param sigma: model parameter (given)
    :return: value of initial condition
    """
    return z * norm.pdf(x, loc=lc, scale=sigma)


def run_fipy_solver(z, sigma, D=1., timeStepDuration=1e-1, nx=100, lc=50, dx=1, xminval=0., xmaxval=100., dataminval=0.,
                    datamaxval=2., resval=1e-6,title="solution",filename="solution.png"):
    """
    Function to solve the PDE of interest using FiPy w/ periodic BC
    :param z: dummy variable
    :param sigma: model parameter (given)
    :param D: diffusion coefficient
    :param timeStepDuration: time step size
    :param nx: number of solution points in mesh
    :param dx: grid spacing
    :param xminval: x axis min for plot
    :param xmaxval: x axis max for plot
    :param dataminval: y axis min for plot
    :param datamaxval: y axis max for plot
    :param resval: max value of residual
    :param title: title for plot
    :param filename: title for plot file (will add png extension)
    :return:
    """
    mesh = meshes.PeriodicGrid1D(nx=nx, dx=dx)  # create grid with periodic boundary conditions
    x = mesh.x  # get x variable from mesh
    u = CellVariable(name="solution variable",
                     mesh=mesh,
                     value=u0(x, z, lc, sigma), hasOld=1)  # initialize solution variable u
    eq = TransientTerm() == DiffusionTerm(coeff=D) - u + u * u # set form of PDE
    viewer = Matplotlib1DViewer(vars=u, title=title,limits={'xmin': xminval, 'xmax': xmaxval}, datamin=dataminval, datamax=datamaxval) # initialize plot
    # u.setValue(u0(x, z, sigma)) # set initial value of u
    # print(u)
    timecount=0
    TSVViewer(vars=u).plot(filename="fipy_output_tsv/soln_data_time_{0}.tsv".format(timecount)) # save initial values
    res = 1e+10 # initial residual (can make smaller)
    while res > resval: # while res greater than max residual value (or timecount < X to set number)
        res = eq.sweep(var=u,
                       dt=timeStepDuration) # sweeping procedure for non-linear PDE
        # print("Residual = %f." % abs(res))
        timecount = timecount + 1
        TSVViewer(vars=u).plot(filename="fipy_output_tsv/soln_data_time_{0}.tsv".format(timecount)) # save value at time step

    viewer.plot(filename=filename) # save plot
    # TSVViewer(vars=u).plot(filename="myTSV.tsv")
    # print(u)

def main():
    sigma = 1
    zvals = [1]#np.linspace(start=1,stop=1,num=1)#20,num=80)
    for z in zvals:
        run_fipy_solver(z, sigma,title="Solution for z={0}".format(z),filename="fipy_output/soln_{0}_{1}.png".format(round(z,5),sigma))

if __name__ == '__main__':
    main()
