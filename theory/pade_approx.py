import numpy as np
from scipy.interpolate import pade
from scipy.interpolate import approximate_taylor_polynomial
from scipy.signal import residue
import pandas as pd

# def get_residues(p,q,m,n):
#     # print(len(coefs))
#     # print(m)
#     # print(n)
#     # p, q = pade(coefs,m,n)
#     rs, pl, k = residue(p,q)
#
#     # print(pl)
#     pole = pl[np.argmin(min(abs(pl)))]
#     res = rs[np.argmin(pole)]
#     return pole, res
#     # return rs, pl

def calc_pade_table(coefs):
    mvals = []
    nvals = []
    pvals = []
    qvals = []
    for m in range(0,len(coefs)):
        for n in range(0,len(coefs)-m):
            p, q = pade(coefs,m,n)
            pvals.append(p.coefficients)
            qvals.append(q.coefficients)
            mvals.append(m)
            nvals.append(n)
    pade_tab = pd.DataFrame(list(zip(mvals,nvals,pvals,qvals)),columns=['m','n','p','q'])
    return pade_tab

def main():
    data = pd.read_csv('spatial_integrals.csv')
    sigma = 1.010900900900901
    coefs_sigma = [1]
    for i in range(3):
        coefs_sigma.append(data.loc[data["sigma"]==sigma,['u2_GQ','u3_GQ','u4_GQ']].values.tolist()[0][i])
    res = calc_pade_table(coefs_sigma)
    #print(np.poly1d(res.iloc[1]['p'])) this is how to get polynomials back
    print(res)

if __name__ == '__main__':
    main()