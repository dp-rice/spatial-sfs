from scipy import integrate
from scipy import special
from scipy.stats import qmc
from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd

def h1(x,sigma):
    """
    function evaluated in Monte Carlo integration for u2
    """
    temp = (4*np.pi*np.pi*(x/sigma)*(x/sigma)+2)*sigma
    return(1/temp)

def integrand1(x,sigma):
    """
    integrand to be evaluated by quadrature for u2
    """
    return(h1(x,sigma)*math.exp(-x*x))

def h2(y,x,sigma):
    """
    function evaluated in Monte Carlo integration for u3
    """
    temp = sigma*sigma*(4*np.pi*np.pi*(y/sigma)*(y/sigma)+4*np.pi*np.pi*(y/sigma)*(y/sigma)+2)*(4*np.pi*np.pi*(-(x/sigma)-(y/sigma))*(-(x/sigma)-(y/sigma))+4*(np.pi*np.pi*(y/sigma)*(y/sigma))+4*np.pi*np.pi*(x/sigma)*(x/sigma)+1)
    return(np.exp(-x*y)/temp)

def integrand2(y,x,sigma): # note - this is in terms of y=\xi'' and x=\xi' to avoid overflow errors
    """
    integrand to be evaluated by quadrature for u3
    """
    num = math.exp(-0.5*sigma*sigma*(-x-y)*(-x-y))*math.exp(-0.5*sigma*sigma*x*x)*math.exp(-0.5*sigma*sigma*y*y)
    denom = (4*np.pi*np.pi*y*y+4*np.pi*np.pi*y*y+2)*(4*np.pi*np.pi*(-x-y)*(-x-y)+4*np.pi*np.pi*y*y+4*np.pi*np.pi*x*x+1)
    return(num/denom)

def h3(z,y,x,sigma):
    """
    LHS function evaluated in Monte Carlo integration for u4
    """
    temp=sigma*sigma*sigma*(4*np.pi*np.pi*(z/sigma)*(z/sigma)+4*np.pi*np.pi*(z/sigma)*(z/sigma)+2)*(4*np.pi*np.pi*(z/sigma)*(z/sigma)+4*np.pi*np.pi*(y/sigma)*(y/sigma)+4*np.pi*np.pi*(-(y/sigma)-(z/sigma))*(-(y/sigma)-(z/sigma))+3)*(4*np.pi*np.pi*(-(x/sigma)-(y/sigma)-(z/sigma))*(-(x/sigma)-(y/sigma)-(z/sigma))+4*np.pi*np.pi*(x/sigma)*(x/sigma)+4*np.pi*np.pi*(y/sigma)*(y/sigma)+4*np.pi*np.pi*(z/sigma)*(z/sigma)+4)
    return(np.exp(-x*y)*np.exp(-x*z)*np.exp(-y*z)/temp)

def integrand3(z,y,x,sigma): # note - this is in terms of z=\xi''', y=\xi'', and x=\xi' to avoid overflow errors
    """
    LHS integrand to be evaluated by quadrature for u4
    """
    num = math.exp(-0.5 * sigma * sigma * (-x - y - z) * (-x - y - z)) * math.exp(
        -0.5 * sigma * sigma * x * x) * math.exp(-0.5 * sigma * sigma * y * y) * math.exp(-0.5 * sigma * sigma * z * z)
    denom = (4*np.pi*np.pi*z*z+4*np.pi*np.pi*z*z+2)*(4*np.pi*np.pi*z*z+4*np.pi*np.pi*y*y+4*np.pi*np.pi*(-y-z)*(-y-z)+3)*(4*np.pi*np.pi*(-x-y-z)*(-x-y-z)+4*np.pi*np.pi*x*x+4*np.pi*np.pi*y*y+4*np.pi*np.pi*z*z+4)
    return(num/denom)

def h4(z,y,x,sigma):
    """
    RHS function evaluated in Monte Carlo integration for u4
    """
    denom1 = sigma*sigma*sigma*(4 * np.pi * np.pi * ((y / sigma) + (z / sigma)) * ((y / sigma) + (z / sigma)) + 4 * np.pi * np.pi * (
                -(y / sigma) - (z / sigma)) * (-(y / sigma) - (z / sigma)) + 2) * (
                         4 * np.pi * np.pi * ((y / sigma) + (z / sigma)) * (
                             (y / sigma) + (z / sigma)) + 4 * np.pi * np.pi * (x / sigma) * (x / sigma) + 3) * (
                         4 * np.pi * np.pi * (z / sigma) * (z / sigma) - 4 * np.pi * np.pi*(
                     (y / sigma) + (z / sigma)) * ((y / sigma) + (z / sigma)) + 1)
    denom2 = sigma*sigma*sigma*(4 * np.pi * np.pi * (z / sigma) * (z / sigma) + 4 * np.pi * np.pi * (-(y / sigma) - (z / sigma)) * (
                -(y / sigma) - (z / sigma)) + 3) * (
                         4 * np.pi * np.pi * (z / sigma) * (z / sigma) + 4 * np.pi * np.pi * (x / sigma) * (
                             x / sigma)+4) * (4 * np.pi * np.pi * ((y / sigma) + (z / sigma)) * (
                (y / sigma) + (z / sigma)) - 4 * np.pi * np.pi*(z / sigma) * (z / sigma) - 1)
    return(np.exp(-x*y)*np.exp(-x*z)*np.exp(-y*z)*((1/denom1)+(1/denom2)))

def integrand4(z,y,x,sigma): # note - this is in terms of z=\xi'''', y=\xi'', and x=\xi' to avoid overflow errors
    """
    RHS integrand to be evaluated by quadrature for u4
    """
    num = math.exp(-0.5*sigma*sigma*(-x-y)*(-x-y))*math.exp(-0.5*sigma*sigma*x*x)*math.exp(-0.5*sigma*sigma*(y-z)*(y-z))*math.exp(-0.5*sigma*sigma*z*z)
    denom1 = (4*np.pi*np.pi*y*y+4*np.pi*np.pi*y*y+2)*(4*np.pi*np.pi*y*y+4*np.pi*np.pi*x*x+3)*(4*np.pi*np.pi*z*z-4*np.pi*np.pi*y*y+1)
    denom2 = (4*np.pi*np.pi*z*z+4*np.pi*np.pi*y*y+3)*(4*np.pi*np.pi*z*z+4*np.pi*np.pi*x*x+4)*(4*np.pi*np.pi*y*y-4*np.pi*np.pi*z*z-1)
    return (num * ((1 / denom1) + (1 / denom2)))

def montecarlo_integral(sigma,dim,pow,func1,func2=None):
    """
    function to perform Monte Carlo integration
    :param sigma: dispersion of sampling kernel f
    :param dim: dimension of integral to be calculated (1, 2, or 3)
    :param pow: parameter for quasi-Monte Carlo, will draw 2**pow samples
    :param func1: first function to be evaluated
    :param func2: second function to be evaluated (used only in u4/dim=3 case here)
    :return: Monte Carlo integration result
    """
    engine = qmc.MultivariateNormalQMC(mean=np.repeat(0, dim), cov=np.diag(np.repeat(0.5, dim)))
    samps = engine.random(2 ** pow)
    if dim==1:
        samps_func = [func1(x, sigma) for x in samps]
        return (math.sqrt(np.pi) * np.mean(list(samps_func)))
    if dim==2:
        samps_func = [func1(y, x, sigma) for (x, y) in samps]
        return(np.pi * np.mean(list(samps_func)))
    if dim==3:
        samps_func1 = [func1(z,y,x,sigma) for (x,y,z) in samps]
        samps_func2 = [func2(z,y,x,sigma) for (x,y,z) in samps]
        return(4*np.pi*math.sqrt(np.pi)*np.mean(list(samps_func1))+np.pi*math.sqrt(np.pi)*np.mean(list(samps_func2)))

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
        return(4*integrate.tplquad(integrandA, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf, lambda x,y: -np.inf, lambda x,y: np.inf,args=(sigma,))[0]
               + integrate.tplquad(integrandB, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf, lambda x,y: -np.inf, lambda x,y: np.inf,args=(sigma,))[0])

def gausshermitequad_integral(n,sigma,func): # works for 1d only
    """
    function to perform Gauss-Hermite quadrature for u2
    :param n: number of sample points to use
    :param sigma: dispersion of sampling kernel f
    :param func: function to integrate
    :return: Gauss-Hermite Quadrature result
    """
    xvals, weights = special.roots_hermite(n)
    series_func = [func(xvals[i], sigma) * weights[i] for i in range(n)]
    return (sum(series_func))

def main():
    sigma_list = np.linspace(1e-8, 100, 1000)
    u2_list = [montecarlo_integral(sigma=s,pow=15,func1=h1,dim=1) for s in sigma_list]
    u2_gh_list = [gausshermitequad_integral(n=1500,sigma=s,func=h1) for s in sigma_list]
    u2_gauss_list = [gaussquad_integral(sigma=s,dim=1,integrandA=integrand1) for s in sigma_list]
    u3_list = [montecarlo_integral(sigma=s, pow=15,func1=h2,dim=2) for s in sigma_list]
    u3_gauss_list = [gaussquad_integral(sigma=s,dim=2,integrandA=integrand2) for s in sigma_list]
    u4_list = [montecarlo_integral(sigma=s,pow=15,func1=h3,func2=h4,dim=3) for s in sigma_list]
    u4_gauss_list =  [gaussquad_integral(sigma=s,dim=3,integrandA=integrand3,integrandB=integrand4) for s in sigma_list]

    df = pd.DataFrame(list(zip(sigma_list, u2_list,u2_gh_list,u2_gauss_list,u3_list,u3_gauss_list,u4_list,u4_gauss_list)),
                      columns=['sigma', 'u2_MC','u2_GH','u2_GQ','u3_MC','u3_GQ','u4_MC','u4_GQ'])

    df.to_csv('spatial_integrals.csv',index=False)

    # # plot for u1 - all methods
    # plt.figure(1)
    # plt.plot(sigma_list,u1_list,label="u1_ev")
    # plt.plot(sigma_list, u1_gh_list,':', label="u1_gh")
    # plt.plot(sigma_list, u1_gauss_list,'--', label="u1_quad")
    # plt.legend()
    # plt.xlabel("spatial dispersion of sample (sigma)")
    # plt.ylabel("value")
    # plt.savefig('u1_allmethods.png')
    #
    # # plot for u2 - all methods
    # plt.figure(2)
    # plt.plot(sigma_list, u2_list, label="u2_ev")
    # plt.plot(sigma_list, u2_gauss_list, '--', label="u2_quad")
    # plt.legend()
    # plt.xlabel("spatial dispersion of sample (sigma)")
    # plt.ylabel("value")
    # plt.savefig('u2_allmethods.png')
    #
    # # plot for u3 - all methods
    # plt.figure(3)
    # plt.plot(sigma_list, u3_list, label="u3_ev")
    # plt.plot(sigma_list, u3_gauss_list, '--', label="u3_quad")
    # plt.legend()
    # plt.xlabel("spatial dispersion of sample (sigma)")
    # plt.ylabel("value")
    # plt.savefig('u3_allmethods.png')
    #
    # # plot for u1, u2, & u3 - monte carlo only
    # plt.figure(4)
    # plt.plot(sigma_list, u1_list, label="u1_ev")
    # plt.plot(sigma_list, u2_list, label="u2_ev")
    # plt.plot(sigma_list, u3_list, label="u3_ev")
    # plt.legend()
    # plt.xlabel("spatial dispersion of sample (sigma)")
    # plt.ylabel("value")
    # plt.savefig("u1u2u3_montecarlo.png")

if __name__ == '__main__':
    main()