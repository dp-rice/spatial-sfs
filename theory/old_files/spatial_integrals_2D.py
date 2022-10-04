from scipy import integrate
import numpy as np
import math
import pandas as pd
from mpi4py import MPI

def integrand5(x1,x2,sigma): # x1=\xi_1',x2=\xi_2'
    """
    integrand to be evaluated by quadrature for u2 in 2D case
    """
    num = math.exp(-0.5 * sigma * sigma * (x1*x1 + x2*x2)) * math.exp(-0.5 * sigma * sigma * (x1*x1 + x2*x2))
    denom = (8 * np.pi * np.pi * x1 * x1) + (8 * np.pi * np.pi * x2 * x2) + 2
    return (num / denom)

def integrand6(y1,y2,x1,x2,sigma): #x1=\xi_1', x2=\xi_2', y_1=\xi_1'',y_2=\xi_2''
    """
    integrand to be evaluated by quadrature for u3 in 2D case
    """
    num = math.exp(-0.5*sigma*sigma*((-x1-y1)*(-x1-y1)+(-x2-y2)*(-x2-y2)))*math.exp(-0.5*sigma*sigma*(y1*y1+y2*y2))*math.exp(-0.5*sigma*sigma*(x1*x1+x2*x2))
    denom = (8*np.pi*np.pi*y1*y1+8*np.pi*np.pi*y2*y2+2)*(4*np.pi*np.pi*x1*x1+4*np.pi*np.pi*x2*x2+4*np.pi*np.pi*y1*y1+4*np.pi*np.pi*y2*y2+4*np.pi*np.pi*(-x1-y1)*(-x1-y1)+4*np.pi*np.pi*(-x2-y2)*(-x2-y2)+3)
    return (num / denom)


def gaussquad_integral(sigma,dim,integrandA,integrandB=None):
    """
    function to perform Gaussian quadrature
    :param sigma: dispersion of sampling kernel f
    :param dim: dimension of integral to be calculated (1, 2, or 3)
    :param integrandA: first integrand to be evaluated
    :param integrandB: second integrand to be evaluated (used only in u4/dim=3 case here)
    :return: Gaussian Quadrature result
    """
    if dim==1:
        return(integrate.quad(integrandA, -np.inf, np.inf, args=(sigma,))[0])
    if dim==2:
        return(integrate.dblquad(integrandA, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf, args=(sigma,))[0])
    if dim==3:
        if integrandB is not None:
            return(integrate.tplquad(integrandA, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf, lambda x,y: -np.inf, lambda x,y: np.inf,args=(sigma,))[0]
                   + integrate.tplquad(integrandB, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf, lambda x,y: -np.inf, lambda x,y: np.inf,args=(sigma,))[0])
        else:
            return (integrate.tplquad(integrandA, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf, lambda x, y: -np.inf,
                              lambda x, y: np.inf, args=(sigma,))[0])
    if dim==4:
        return(integrate.nquad(func=integrandA,ranges=[(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)],args=(sigma,))[0])

def main():

    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    # sigma_list = np.linspace(1e-2, 100, 50)

    # rank=0
    sigma_list = []

    if rank == 0:
        sigma_list = np.linspace(1e-2, 100, 50)
        sigma_list_tuple = [[sigma_list[i:i+10]] for i in range(0,len(sigma_list),10)]
        # print(sigma_list)
        # print(sigma_list_tuple)

    sigma_list = comm.scatter(sigma_list,root=0)

    u3_gauss_list = []
    for item in sigma_list:
        u3_gauss_list.append(gaussquad_integral(sigma=item,dim=4,integrandA=integrand6))

    new_u3_list = comm.gather(u3_gauss_list,root=0)

    # u3_gauss_list = [gaussquad_integral(sigma=s, dim=4, integrandA=integrand6) for s in sigma_list]
    df = pd.DataFrame(
        list(zip(sigma_list,new_u3_list)),
        columns=['sigma','u3_GQ'])
    df.to_csv('spatial_integrals_2D_u3.csv', index=False)


if __name__ == '__main__':
    main()