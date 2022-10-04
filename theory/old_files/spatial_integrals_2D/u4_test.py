from scipy import integrate
import numpy as np
import math
import argparse

def integrand_u4_LHS(z1,z2,y1,y2,x1,x2,sigma): # note - this is in terms of z=\xi''', y=\xi'', and x=\xi'
    """
    LHS integrand to be evaluated by quadrature for u4 in 2D case (formerly integrand3)
    """
    num = 4*math.exp(-0.5 * sigma * sigma * ((-x1 - y1 - z1) * (-x1 - y1 - z1)+(-x2 - y2 - z2) * (-x2 - y2 - z2))) * math.exp(-0.5 * sigma * sigma * ((x1 * x1)+(x2*x2))) * math.exp(-0.5 * sigma * sigma * (y1 * y1+y2*y2)) * math.exp(-0.5 * sigma * sigma * (z1 * z1+z2*z2))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, help="value of sigma parameter")
    args = parser.parse_args()
    options={'epsrel':1e-6,'epsabs':1e-6}
    u4_temp = u4_gauss_list.append(gaussquad_integral(sigma=s,dim=6,integrandA=integrand_u4_LHS,integrandB=integrand_u4_RHS,opts=[options,options,options,options,options,options]))
    filename = "u4_sigma{}.txt".format(sigma)
    f = open(filename,"w")
    f.write(str(u4_temp))
    f.close()

if __name__ == '__main__':
    main()
