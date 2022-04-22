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
    mult = []
    if len(pl)>0:
        pl_abs = [abs(x) for x in pl]
        pole = pl[np.argmin(pl_abs)]
        res = rs[np.argmin(pl_abs)]
        mult_temp = len([x for x in pl if x==pole])
        mult.append(mult_temp)
    else:
        pole = np.nan
        res = np.nan
    return pole, res, mult


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
    multvals = []
    # tab = tab.reset_index()
    for row in tab.itertuples():
        p, q = get_pade_poly(tab,row[tab.columns.get_loc('m')+1],row[tab.columns.get_loc('n')+1])
        pole, res, mult = get_residues(p, q)
        polevals.append(pole)
        resvals.append(res)
        multvals.append(mult)
    tab_new = tab
    tab_new['pole'] = polevals
    tab_new['residue'] = resvals
    tab_new['multiplicity_pole'] = multvals
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
    for row in tab.itertuples():
        err = []
        tc = row[5]
        while len(tc)<len(coefs): # make lists the same length
            tc.append(0)
        err = [a-b for a,b in zip(coefs, tc)]
        err_vals.append(err)
        coefs_vals.append(coefs)
    tab_new = tab
    tab_new['coef_input_vals'] = coefs_vals
    tab_new['error_vals'] = err_vals
    return tab_new

def main():
    # data = pd.read_csv('spatial_integrals.csv')
    # sigma_list = data['sigma'].tolist()
    # res = pd.DataFrame()
    # for j in range(len(sigma_list)):
    #     coefs_sigma = [1]
    #     sigma = sigma_list[j]
    #     for i in range(3):
    #         coefs_sigma.append(data.loc[data["sigma"]==sigma,['u2_GQ','u3_GQ','u4_GQ']].values.tolist()[0][i])
    #     temp = calc_pade_table(coefs_sigma)
    #     temp = calcError(temp,coefs_sigma)
    #     temp = calc_pole_res(temp)
    #     temp.insert(loc=0, column='sigma', value=np.repeat(sigma,temp.shape[0]))
    #     res = pd.concat([res, temp], ignore_index=True, sort=False)
    res = pd.DataFrame()
    coefs = np.repeat(1,10).tolist()
    print(coefs)
    res = calc_pade_table(coefs)
    res = calcError(res,coefs)
    res = calc_pole_res(res)

    res.to_csv('pade_approx_test.csv', index=False)

if __name__ == '__main__':
    main()