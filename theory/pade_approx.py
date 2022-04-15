import numpy as np
import numpy.linalg
from scipy.interpolate import pade
from scipy.interpolate import approximate_taylor_polynomial
from scipy.signal import residue
import pandas as pd

def get_residues(p,q):
    """
    Given function p/q, returns pole closest to zero and its residue
    """
    rs, pl, k = residue(p,q)
    if len(pl)>0:
        pole = pl[np.argmin(min(abs(pl)))]
        res = rs[np.argmin(pole)]
    else:
        pole = np.nan
        res = np.nan
    return pole, res


def calc_pade_table(coefs):
    """
    Creates table containing coefficients for Pade approximation
    given coefficients (coefs) over all possible pairs (m,n) and
    associated Taylor series coefficients
    """
    mvals = []
    nvals = []
    pvals = []
    qvals = []
    taylor_coefs = []
    for m in range(0,len(coefs)):
        for n in range(0,len(coefs)-m):
            try:
                p, q = pade(coefs,m,n)
                pvals.append(p.coefficients)
                qvals.append(q.coefficients)
                mvals.append(m)
                nvals.append(n)
                def f(x):
                    return p(x)/q(x)
                taylor = approximate_taylor_polynomial(f,0,len(coefs)-1,1).coefficients.tolist()
                taylor.reverse()
                taylor_coefs.append(taylor)
            except numpy.linalg.LinAlgError as err:
                print("Warning: error thrown for m = %s and n = %s.\n %s" % (m,n,err))

    pade_tab = pd.DataFrame(list(zip(mvals,nvals,pvals,qvals,taylor_coefs)),columns=['m','n','p','q','tc'])
    return pade_tab

def calc_pole_res(tab):
    """
    Calculates pole nearest to zero and its residue for each Pade
    approximant given in tab and appends to data frame
    """
    polevals = []
    resvals = []
    # tab = tab.reset_index()
    for row in tab.itertuples():
        p, q = get_pade_poly(tab,row[tab.columns.get_loc('m')+1],row[tab.columns.get_loc('n')+1])
        pole, res = get_residues(p, q)
        polevals.append(pole)
        resvals.append(res)
    tab_new = tab
    tab_new['pole'] = polevals
    tab_new['residue'] = resvals
    return tab_new

def get_pade_poly(tab,m,n):
    """
    Returns numpy polynomial object for a given entry in tab
    """
    tab_sub = tab[(tab['m'] == m) & (tab['n'] ==n )]
    p = np.poly1d(tab_sub.iloc[0]['p'])
    q = np.poly1d(tab_sub.iloc[0]['q'])
    return p,q

def calcError(tab,coefs):
    """
    Calculates error between unused coefficients (cumulants) and Talor coefficients
    of Pade approx
    """
    # tab = tab.reset_index()
    err_vals = []
    for row in tab.itertuples():
        err = []
        tc = row[tab.columns.get_loc('tc')].tolist()
        while len(tc)<len(coefs): # make lists the same length
            tc.append(0)
        for i in range(len(tc)):
            err.append(coefs[i]-tc[i])
        err_vals.append(err)
    tab_new = tab
    tab_new['error_vals'] = err_vals
    return tab_new

def main():
    data = pd.read_csv('spatial_integrals.csv')
    sigma = 1.010900900900901 # eventually want to iterate over sigma values and save to csv
    coefs_sigma = [1]
    for i in range(3):
        coefs_sigma.append(data.loc[data["sigma"]==sigma,['u2_GQ','u3_GQ','u4_GQ']].values.tolist()[0][i])
    # coefs_sigma = [1,-1,1,-1] test case
    res = calc_pade_table(coefs_sigma)
    res = calcError(res,coefs_sigma)
    res = calc_pole_res(res)
    print(res)
    res.to_csv('pade_approx.csv', index=False)

if __name__ == '__main__':
    main()