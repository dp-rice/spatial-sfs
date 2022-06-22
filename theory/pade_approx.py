import numpy as np
import numpy.linalg
from scipy.interpolate import pade
# from scipy.interpolate import approximate_taylor_polynomial
from scipy.signal import residue
import pandas as pd
import math
import random

def deriv_rational(p,q,n=1):
    """
    Generate nth derivative of rational p/q using quotient rule
    """
    p = np.poly1d(p)
    q = np.poly1d(q)
    for i in range(n):
        p_der = np.polyder(p)
        q_der = np.polyder(q)
        new_p = np.polysub(np.polymul(q,p_der),np.polymul(p,q_der))
        new_q = np.polymul(q,q)
        p = new_p
        q = new_q
    return p,q


def get_taylor_coefs(p,q,m,a=0):
    """
    Compute Taylor series coefficients for rational function p/q
    """
    coefs = []
    for i in range(m):
        p_i, q_i = deriv_rational(p=p,q=q,n=i)
        coefs.append((p_i(a)/q_i(a))/math.factorial(i))
    return coefs


def get_residues(p,q):
    """
    Given function p/q, returns pole closest to zero and its residue
    """
    rs, pl, k = residue(p,q)
    mult = []
    pl_all = []
    if len(pl)>0:
        pl_abs = [abs(x) for x in pl]
        print(pl)
        print(pl_abs)
        pole = pl[np.argmin(pl_abs)]
        print(pole)
        res = rs[np.argmin(pl_abs)]
        mult_temp = len([x for x in pl if x==pole])
        mult.append(mult_temp)
        pl_all.append(pl)
    else:
        pole = np.nan
        res = np.nan
    return pole, res, mult, pl_all


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
                # taylor = approximate_taylor_polynomial(f,0,len(coefs)-1,1).coefficients.tolist()
                # taylor.reverse()
                taylor = get_taylor_coefs(p,q,len(coefs))
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
    multvals = []
    plvals = []
    # tab = tab.reset_index()
    for row in tab.itertuples():
        p, q = get_pade_poly(tab,row[tab.columns.get_loc('m')+1],row[tab.columns.get_loc('n')+1])
        pole, res, mult, pl_all = get_residues(p, q)
        print("here")
        print(row[tab.columns.get_loc('m')+1])
        print(row[tab.columns.get_loc('n') + 1])
        print(pole)
        polevals.append(pole)
        resvals.append(res)
        multvals.append(mult)
        plvals.append(pl_all)
    tab_new = tab
    tab_new['pole'] = polevals
    tab_new['residue'] = resvals
    tab_new['multiplicity_pole'] = multvals
    tab_new['poles_all'] = plvals
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
    coefs_vals = []
    err_next_vals = []
    for row in tab.itertuples():
        err = []
        # print(row)
        tc = row[5]
        m = row[1]
        n = row[2]
        while len(tc)<len(coefs): # make lists the same length
            tc.append(0)
        err = [a-b for a,b in zip(coefs, tc)]
        err_vals.append(err)
        coefs_vals.append(coefs)
        err_next_vals.append(err[m+n])
    tab_new = tab
    tab_new['coef_input_vals'] = coefs_vals
    tab_new['error_vals_all'] = err_vals
    tab_new['error_next'] = err_next_vals
    return tab_new

def main():
    data = pd.read_csv('spatial_integrals.csv')
    sigma_list = data['sigma'].tolist()
    res = pd.DataFrame()
    for j in range(len(sigma_list)):
        coefs_sigma = [1]
        sigma = sigma_list[j]
        for i in range(3):
            coefs_sigma.append(data.loc[data["sigma"]==sigma,['u2_GQ','u3_GQ','u4_GQ']].values.tolist()[0][i])
        # coefs_sigma = [x+random.uniform(-5e-8,5e-8) for x in coefs_sigma]
        # print(coefs_sigma)
        temp = calc_pade_table(coefs_sigma)
        temp = calcError(temp,coefs_sigma)
        temp = calc_pole_res(temp)
        temp.insert(loc=0, column='sigma', value=np.repeat(sigma,temp.shape[0]))
        res = pd.concat([res, temp], ignore_index=True, sort=False)

    # print(get_taylor_coefs(np.poly1d([1,1,0]),np.poly1d([-1,1]),5))
    # res = pd.DataFrame()
    # coefs = np.repeat(1,10).tolist()
    # print(coefs)
    # res = calc_pade_table(coefs)
    # res = calcError(res,coefs)
    # res = calc_pole_res(res)
    res.to_csv('pade_approx.csv', index=False)

if __name__ == '__main__':
    main()