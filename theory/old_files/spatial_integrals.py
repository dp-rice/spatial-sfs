from scipy import integrate
# from scipy import special
# from scipy.stats import qmc
import numpy as np
import math
import pandas as pd
import time

def integrand1(x,sigma): #  x=\xi' notation
    """
    integrand to be evaluated by quadrature for u2
    """
    num = math.exp(-0.5*sigma*sigma*x*x)*math.exp(-0.5*sigma*sigma*x*x)
    denom = 8*np.pi*np.pi*x*x+2
    return(num/denom)

def integrand2(y,x,sigma): # note - this is in terms of y=\xi'' and x=\xi'
    """
    integrand to be evaluated by quadrature for u3
    """
    num = math.exp(-0.5*sigma*sigma*(-x-y)*(-x-y))*math.exp(-0.5*sigma*sigma*x*x)*math.exp(-0.5*sigma*sigma*y*y)
    denom = (8*np.pi*np.pi*y*y+2)*(4*np.pi*np.pi*(-x-y)*(-x-y)+4*np.pi*np.pi*y*y+4*np.pi*np.pi*x*x+3)
    return(2*num/denom)

def integrand3(z,y,x,sigma): # note - this is in terms of z=\xi''', y=\xi'', and x=\xi'
    """
    LHS integrand to be evaluated by quadrature for u4
    """
    num = 4*math.exp(-0.5 * sigma * sigma * (-x - y - z) * (-x - y - z)) * math.exp(
        -0.5 * sigma * sigma * x * x) * math.exp(-0.5 * sigma * sigma * y * y) * math.exp(-0.5 * sigma * sigma * z * z)
    denom = (8*np.pi*np.pi*z*z+2)*(4*np.pi*np.pi*z*z+4*np.pi*np.pi*y*y+4*np.pi*np.pi*(-y-z)*(-y-z)+3)*(4*np.pi*np.pi*(-x-y-z)*(-x-y-z)+4*np.pi*np.pi*x*x+4*np.pi*np.pi*y*y+4*np.pi*np.pi*z*z+4)
    return(num/denom)

def integrand4(z,y,x,sigma): # note - this is in terms of z=\xi'''', y=\xi'', and x=\xi'
    """
    RHS integrand to be evaluated by quadrature for u4
    """
    num = math.exp(-0.5*sigma*sigma*(-x-z)*(-x-z))*math.exp(-0.5*sigma*sigma*x*x)*math.exp(-0.5*sigma*sigma*(z-y)*(z-y))*math.exp(-0.5*sigma*sigma*y*y)
    denom1 = (8*np.pi*np.pi*z*z+2)*(4*np.pi*np.pi*x*x+4*np.pi*np.pi*z*z+4*np.pi*np.pi*(-x-z)*(-x-z)+3)*(4*np.pi*np.pi*y*y+4*np.pi*np.pi*(z-y)*(z-y)-4*np.pi*np.pi*z*z+1)
    denom2 = (4*np.pi*np.pi*z*z+4*np.pi*np.pi*y*y+4*np.pi*np.pi*(z-y)*(z-y)+3)*(4*np.pi*np.pi*y*y+4*np.pi*np.pi*(z-y)*(z-y)+4*np.pi*np.pi*x*x+4*np.pi*np.pi*(-x-z)*(-x-z)+4)*(4*np.pi*np.pi*z*z-4*np.pi*np.pi*y*y-4*np.pi*np.pi*(z-y)*(z-y)-1)
    try:
        ans = num * ((1 / denom1) + (1 / denom2))
    except ZeroDivisionError:
        ans = 0
        print("error: division by zero")
    return ans

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

# def montecarlo_integral(sigma,dim,pow,func1,func2=None):
#     """
#     function to perform Monte Carlo integration
#     :param sigma: dispersion of sampling kernel f
#     :param dim: dimension of integral to be calculated (1, 2, or 3)
#     :param pow: parameter for quasi-Monte Carlo, will draw 2**pow samples
#     :param func1: first function to be evaluated
#     :param func2: second function to be evaluated (used only in u4/dim=3 case here)
#     :return: Monte Carlo integration result
#     """
#     engine = qmc.MultivariateNormalQMC(mean=np.repeat(0, dim), cov=np.diag(np.repeat(0.5, dim)))
#     samps = engine.random(2 ** pow)
#     if dim==1:
#         samps_func = [func1(x, sigma) for x in samps]
#         return (math.sqrt(np.pi) * np.mean(list(samps_func)))
#     if dim==2:
#         samps_func = [func1(y, x, sigma) for (x, y) in samps]
#         return(np.pi * np.mean(list(samps_func)))
#     if dim==3:
#         samps_func1 = [func1(z,y,x,sigma) for (x,y,z) in samps]
#         samps_func2 = [func2(z,y,x,sigma) for (x,y,z) in samps]
#         return(4*np.pi*math.sqrt(np.pi)*np.mean(list(samps_func1))+np.pi*math.sqrt(np.pi)*np.mean(list(samps_func2)))

# def gausshermitequad_integral(n,sigma,func): # works for 1d only
#     """
#     function to perform Gauss-Hermite quadrature for u2
#     :param n: number of sample points to use
#     :param sigma: dispersion of sampling kernel f
#     :param func: function to integrate
#     :return: Gauss-Hermite Quadrature result
#     """
#     xvals, weights = special.roots_hermite(n)
#     series_func = [func(xvals[i], sigma) * weights[i] for i in range(n)]
#     return (sum(series_func))

def main():
    sigma_list = np.linspace(1e-2, 100, 50)
    d="2D"
    if d=="1D":
        print("calculating u2")
        u2_gauss_list = [gaussquad_integral(sigma=s,dim=1,integrandA=integrand1) for s in sigma_list]
        print("calculating u3")
        u3_gauss_list = [gaussquad_integral(sigma=s,dim=2,integrandA=integrand2) for s in sigma_list]
        print("calculating u4")
        u4_gauss_list =  [gaussquad_integral(sigma=s,dim=3,integrandA=integrand3,integrandB=integrand4) for s in sigma_list]
        df = pd.DataFrame(
            list(zip(sigma_list, u2_gauss_list, u3_gauss_list,u4_gauss_list)),
            columns=['sigma', 'u2_GQ', 'u3_GQ','u4_GQ'])
        df.to_csv('spatial_integrals_1D.csv',index=False)
    elif d=="2D":
        print("calculating u2 (2D)")
        u2_gauss_list=[gaussquad_integral(sigma=s,dim=2,integrandA=integrand5) for s in sigma_list]
        #print("calculating u3 (2D)")
        #u3_gauss_list = [gaussquad_integral(sigma=s, dim=4, integrandA=integrand6) for s in sigma_list]
        df = pd.DataFrame(
            list(zip(sigma_list, u2_gauss_list)),
            columns=['sigma', 'u2_GQ'])
        df.to_csv('spatial_integrals_2D_u2.csv', index=False)
    elif d=="test":
        start = time.time()
        #u3_test = gaussquad_integral(sigma=,dim=4,integrandA=integrand6)
        end = time.time()
        print(end - start)
        print(u3_test)

if __name__ == '__main__':
    main()