from scipy import integrate
import numpy as np
import math
import pandas as pd
from mpi4py import MPI

def integrand_u2(x1,x2,sigma): # x1=\xi_1',x2=\xi_2'
    """
    integrand to be evaluated by quadrature for u2 in 2D case
    """
    num = math.exp(-0.5 * sigma * sigma * (x1*x1 + x2*x2)) * math.exp(-0.5 * sigma * sigma * (x1*x1 + x2*x2))
    denom = (8 * np.pi * np.pi * x1 * x1) + (8 * np.pi * np.pi * x2 * x2) + 2
    return (num / denom)

def integrand_u3(y1,y2,x1,x2,sigma): #x1=\xi_1', x2=\xi_2', y_1=\xi_1'',y_2=\xi_2''
    """
    integrand to be evaluated by quadrature for u3 in 2D case
    """
    num = math.exp(-0.5*sigma*sigma*((-x1-y1)*(-x1-y1)+(-x2-y2)*(-x2-y2)))*math.exp(-0.5*sigma*sigma*(y1*y1+y2*y2))*math.exp(-0.5*sigma*sigma*(x1*x1+x2*x2))
    denom = (8*np.pi*np.pi*y1*y1+8*np.pi*np.pi*y2*y2+2)*(4*np.pi*np.pi*x1*x1+4*np.pi*np.pi*x2*x2+4*np.pi*np.pi*y1*y1+4*np.pi*np.pi*y2*y2+4*np.pi*np.pi*(-x1-y1)*(-x1-y1)+4*np.pi*np.pi*(-x2-y2)*(-x2-y2)+3)
    return (num / denom)

def integrand_u4_LHS(z1,z2,y1,y2,x1,x2,sigma): # note - this is in terms of z=\xi''', y=\xi'', and x=\xi'
    """
    LHS integrand to be evaluated by quadrature for u4 in 2D case (formerly integrand3)
    """
    num = 4*math.exp(-0.5 * sigma * sigma * ((-x1 - y1 - z1) * (-x1 - y1 - z1)+(-x2 - y2 - z2) * (-x2 - y2 - z2))) * math.exp(
        -0.5 * sigma * sigma * ((x1 * x1)+(x2*x2)) * math.exp(-0.5 * sigma * sigma * (y1 * y1+y2*y2)) * math.exp(-0.5 * sigma * sigma * (z1 * z1+z2*z2))
    denom = (8*np.pi*np.pi*z1*z1+8*np.pi*np.pi*z2*z2+2)*(4*np.pi*np.pi*z1*z1+4*np.pi*np.pi*z2*z2+4*np.pi*np.pi*y1*y1+4*np.pi*np.pi*y2*y2+4*np.pi*np.pi*(-y1-z1)*(-y1-z1)+4*np.pi*np.pi*(-y2-z2)*(-y2-z2)+3)*(4*np.pi*np.pi*(-x1-y1-z1)*(-x1-y1-z1)+4*np.pi*np.pi*(-x2-y2-z2)*(-x2-y2-z2)+4*np.pi*np.pi*x1*x1+4*np.pi*np.pi*x2*x2+4*np.pi*np.pi*y1*y1+4*np.pi*np.pi*y2*y2+4*np.pi*np.pi*z1*z1+4*np.pi*np.pi*z2*z2+4)
    return(num/denom)

def integrand_u4_RHS(z1,z2,y1,y2,x1,x2,sigma): # note - this is in terms of z=\xi'''', y=\xi'', and x=\xi'
    """
    RHS integrand to be evaluated by quadrature for u4 in 2D casee (formerly integrand4)
    """
    num = math.exp(-0.5*sigma*sigma*((-x1-z1)*(-x1-z1)+(-x2-z2)*(-x2-z2)))*math.exp(-0.5*sigma*sigma*(x1*x1+x2*x2))*math.exp(-0.5*sigma*sigma*((z1-y1)*(z1-y1)+(z2-y2)*(z2-y2)))*math.exp(-0.5*sigma*sigma*(y1*y1+y2*y2))
    denom1 = (8*np.pi*np.pi*z1*z1+8*np.pi*np.pi*z2*z2+2)*(4*np.pi*np.pi*x1*x1+4*np.pi*np.pi*x2*x2+4*np.pi*np.pi*z1*z1+4*np.pi*np.pi*z2*z2+4*np.pi*np.pi*(-x1-z1)*(-x1-z1)+4*np.pi*np.pi*(-x2-z2)*(-x2-z2)+3)*(4*np.pi*np.pi*y1*y1+4*np.pi*np.pi*y2*y2+4*np.pi*np.pi*(z1-y1)*(z1-y1)+4*np.pi*np.pi*(z2-y2)*(z2-y2)-4*np.pi*np.pi*z1*z1-4*np.pi*np.pi*z2*z2+1)
    denom2 = (4*np.pi*np.pi*z1*z1+4*np.pi*np.pi*z2*z2+4*np.pi*np.pi*y1*y1+4*np.pi*np.pi*y2*y2+4*np.pi*np.pi*(z1-y1)*(z1-y1)+4*np.pi*np.pi*(z2-y2)*(z2-y2)+3)*(4*np.pi*np.pi*y1*y1+4*np.pi*np.pi*y2*y2+4*np.pi*np.pi*(z1-y1)*(z1-y1)+4*np.pi*np.pi*(z2-y2)*(z2-y2)+4*np.pi*np.pi*x1*x1+4*np.pi*np.pi*x2*x2+4*np.pi*np.pi*(-x1-z1)*(-x1-z1)+4*np.pi*np.pi*(-x2-z2)*(-x2-z2)+4)*(4*np.pi*np.pi*z1*z1+4*np.pi*np.pi*z2*z2-4*np.pi*np.pi*y1*y1-4*np.pi*np.pi*y2*y2-4*np.pi*np.pi*(z1-y1)*(z1-y1)-4*np.pi*np.pi*(z2-y2)*(z2-y2)-1)
    try:
        ans = num * ((1 / denom1) + (1 / denom2))
    except ZeroDivisionError:
        ans = 0
        print("error: division by zero")
    return ans


def gaussquad_integral(sigma,dim,integrandA,integrandB=None,opts=None):
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
        return(integrate.nquad(func=integrandA,ranges=[(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)],args=(sigma,),opts=opts)[0])
    if dim==6:
        return(integrate.nquad(func=integrandA,ranges=[(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)],args=(sigma,),opts=opts)[0]+integrate.nquad(func=integrandB,ranges=[(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)],args=(sigma,),opts=opts)[0])

def main():

    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    size = comm.Get_size()

    sigma_list = []
    sigma_list_tuple = []
    if rank == 0:
        sigma_list = np.linspace(1e-2,100, 50)
        sigma_list_tuple = [[sigma_list[i:i+5]] for i in range(0,len(sigma_list),5)]
        data = [sigma_list_tuple[x] for x in range(size)]

    else:
        data = None
    
    sigma_list=np.linspace(1e-2,100,50)
    data = comm.scatter(data,root=0)
    u2_gauss_list =[]
    u3_gauss_list = []
    u4_gauss_list = []
    options={'epsrel':1e-6,'epsabs':1e-6}
    for item in data:
        for s in item:
            u2_gauss_list.append(gaussquad_integral(sigma=s, dim=2, integrandA=integrand_u2, opts=[options, options]))
            u3_gauss_list.append(gaussquad_integral(sigma=s,dim=4,integrandA=integrand_u3,opts=[options,options,options,options]))
            u4_gauss_list.append(gaussquad_integral(sigma=s,dim=6,integrandA=integrand_u4_LHS,integrandB=integrand_u4_RHS,opts=[options,options,options,options,options,options]))
    new_u2_list = comm.gather(u2_gauss_list, root=0)
    new_u3_list = comm.gather(u3_gauss_list,root=0)
    new_u4_list = comm.gather(u4_gauss_list, root=0)

    if rank==0:
        print("here")
        new_u2_list_concat = sum(new_u2_list, [])
        new_u3_list_concat = sum(new_u3_list, [])
        new_u4_list_concat = sum(new_u4_list, [])
        df = pd.DataFrame(
            list(zip(sigma_list,new_u2_list_concat,new_u3_list_concat,new_u4_list_concat)),
            columns=['sigma','u2_GQ','u3_GQ','u4_GQ'])
        df.to_csv('spatial_integrals_2D_all.csv', index=False)


if __name__ == '__main__':
    main()
