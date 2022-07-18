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


def run_fipy_solver(z, sigma, D=1., timeStepDuration=1, nx=100, lc=50, dx=1, xminval=0., xmaxval=100., dataminval=-0.5,
                    datamaxval=5, resval=1e-6,title="solution",filename="solution.png"):
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
    TSVViewer(vars=u).plot(filename="fipy_output_tsv_test/soln_data_time_{0}.tsv".format(0)) # save initial values
    viewer.plot(filename="fipy_output_test/soln_data_time_{0}.png".format(0))  # save plot initial
    res = 1e+10 # initial residual (can make smaller)
    steps=20
    for step in range(steps):
        res = 1e+10
        u.updateOld()
        n=0
        while res > resval: # while res greater than max residual value make this 1e-6
            # print(res)
            res = eq.sweep(var=u,dt=timeStepDuration)
            if n>500:
                # print("failed to converge in 500 sweeps")
                break
            n += 1
        if step % 1 == 0:
            TSVViewer(vars=u).plot(filename=filename.replace("time",str(step+1))) # save value at time step

    viewer.plot(filename=filename.replace("time","final"))



def run_fipy_singularities(z_range,sigma, D=1., timeStepDuration=1, nx=100, lc=50, dx=1, resval=1e-2,nsteps=20):
    res = 1e+10 # initial residual (can make smaller)
    steps=nsteps
    zcrit=np.nan
    for z in z_range:
        print(z)
        mesh = meshes.PeriodicGrid1D(nx=nx, dx=dx)  # create grid with periodic boundary conditions
        x = mesh.x  # get x variable from mesh
        u = CellVariable(name="solution variable",
                         mesh=mesh,
                         value=u0(x, z, lc, sigma), hasOld=1)  # initialize solution variable u
        eq = TransientTerm() == DiffusionTerm(coeff=D) - u + u * u  # set form of PDE
        for step in range(steps):
            # print(step)
            res = 1e+10
            u.updateOld()
            n=0
            while res > resval: # while res greater than max residual value
                res = eq.sweep(var=u,dt=timeStepDuration)
                if n>500:
                    # print("failed to converge in 500 sweeps")
                    break
                n += 1
            if u[50]>u0(50,z,lc,sigma): # if not decaying
                zcrit=z
                return zcrit

    return zcrit


def main():
    zvals = np.linspace(start=1,stop=100,num=500)
    sigma_list=[1,2,5,10,15,20,25]
    zcrit_list = [run_fipy_singularities(zvals,s) for s in sigma_list]
    print(zcrit_list)

if __name__ == '__main__':
    main()
