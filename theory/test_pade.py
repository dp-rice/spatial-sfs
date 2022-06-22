import pade_approx
import plots_pade
import pandas as pd
from scipy.interpolate import approximate_taylor_polynomial
import math
import numpy as np

def main():
    # def func(x):
    #     return (np.e**(-x))/(1-x)
    # print(f(0))
    def func(x):
        return np.e**x
    coefs = approximate_taylor_polynomial(f=func, x=0, degree=5, scale=0.01).coefficients.tolist()
    coefs.reverse()
    print(coefs)
    res = pade_approx.calc_pade_table(coefs)
    res = pade_approx.calcError(res,coefs)
    res = pade_approx.calc_pole_res(res)
    res.to_csv('pade_test_exp.csv', index=False)

if __name__ == '__main__':
    main()