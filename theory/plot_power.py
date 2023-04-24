import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.special import erfinv
from scipy.stats import gamma
import sys
import math
sys.path.insert(1, 'numerics_snakemake')
from plot_sfs import *


def power(zs,sigma,s, sigma_vals,res_vals,pole_vals,Nval=1):
    beta = rate_p(sigma,s,sigma_vals,pole_vals,N=Nval)
    alpha = shape_p(sigma,s,sigma_vals,res_vals,N=Nval)
    return(1-gamma.cdf(zs,a=alpha,scale=1/beta))

def calc_zs(b,Vp=vp_val,n=10000,a=0.05):
    vs = 2*erfinv(1-a)**2
    return(0.5 - (0.5)*math.sqrt((1-2*vs*(Vp/n))/(b**2)))

def calc_V_G(kappa_list,s,sel_grad,L=1000,mu=1e-8,d=2,D=1,N=10000):
    V_list = [(2*L*mu*k)/(sel_grad**2*np.sqrt(D/s)**d*N) for k in kappa_list]
    return(V_list)

def calc_V_E():

def calc_V_P:



def main():
    colors = sns.color_palette("colorblind", 3)
    sigma_list = np.logspace(-2, 2, 100)
    data = pd.read_csv(
        "numerics_snakemake/cleaned_data_dim2_errorFalse.csv")  # always use errorFalse version to get higher orders
    data = data.loc[data['poly_type'] == '1_1']

    a = 0.05
    # vs = 2*erfinv(1-a)**2
    # b = 10
    # n = 10000
    # Vp = 1
    # zs = 0.5 - (0.5)*math.sqrt((1-2*vs*(Vp/n))/(b**2))
    # print(zs)

    # zs_vals = [1e-12,1e-10,1e-8]
    # zs_vals = [1e-7,1e-5,1e-3]
    #sel_vals = [0.001,0.01,0.1]
    #zs_vals = [get_zs(s) for s in sel_vals]
    # print(zs_vals)
    #zs_val = 1e-5#get_zs(b=1,a=1e-8) #beta=0.01 #1e-5
    # zs_val=0.2
    #zs_val=0.1
    #zs_val=0.01
    zs_val=0.001
    #zs_val=0.0001
    # zs_val = get_zs(b=1.0001,a=1e-8)
    print(zs_val)
    sel = 0.1
    nvals = [1000,10000,100000]

    fig, axs = plt.subplots(1, 3)
    axs[0].plot(sigma_list,[power(zs_val,sg, sel,data['sigma'],data['residues'],data['poles'],nvals[0]) for sg in sigma_list],
                color=colors[0])
    # axs[0].plot(sigma_list,
    #             [power(zs_val, sg, sel, data['sigma'], data['residues'], data['poles'], nvals[0]) for sg in
    #              sigma_list],color=colors[1])
    # axs[0].plot(sigma_list,
    #             [power(zs_val, sg, sel, data['sigma'], data['residues'], data['poles'], nvals[0]) for sg in
    #              sigma_list],color=colors[2])
    # axs[0].legend(labels=[str(sv) for sv in sel_vals], title="s")
    axs[0].set_title("n=" + str(nvals[0])+",\nZ*="+str(zs_val))
    axs[0].set_xscale("log")
    # axs[0].set_yscale("log")
    axs[0].set_xlabel("sigma")
    axs[0].set_ylabel("value")
    axs[0].set_xlim(1e-2, 1e2)
    axs[0].set_ylim(0, 1)


    axs[1].plot(sigma_list,
                [power(zs_val, sg,sel, data['sigma'], data['residues'], data['poles'], nvals[1]) for sg in sigma_list],color=colors[0])
    # axs[1].plot(sigma_list,
    #             [power(zs_val, sg, sel, data['sigma'], data['residues'], data['poles'], nvals[1]) for sg in
    #              sigma_list],color=colors[1])
    # axs[1].plot(sigma_list,
    #             [power(zs_vals[1], sg, sel_vals[2], data['sigma'], data['residues'], data['poles'], nval) for sg in
    #              sigma_list],color=colors[2])
    # axs[1].legend(labels=[str(sv) for sv in sel_vals], title="s")
    axs[1].set_title("n=" + str(nvals[1])+",\nZ*="+str(zs_val))
    axs[1].set_xscale("log")
    # axs[1].set_yscale("log")
    axs[1].set_xlabel("sigma")
    axs[1].set_ylabel("value")
    axs[1].set_xlim(1e-2, 1e2)
    axs[1].set_ylim(0, 1)

    axs[2].plot(sigma_list,
                [power(zs_val, sg,sel, data['sigma'], data['residues'], data['poles'], nvals[2]) for sg in sigma_list],color=colors[0])
    # axs[2].plot(sigma_list,
    #             [power(zs_vals[1], sg, sel_vals[1], data['sigma'], data['residues'], data['poles'], nval) for sg in
    #              sigma_list],color=colors[1])
    # axs[2].plot(sigma_list,
    #             [power(zs_vals[1], sg, sel_vals[2], data['sigma'], data['residues'], data['poles'], nval) for sg in
    #              sigma_list],color=colors[2])
    # axs[2].legend(labels=[str(sv) for sv in sel_vals], title="s")
    axs[2].set_title("n=" + str(nvals[2])+",\nZ*="+str(zs_val))
    axs[2].set_xscale("log")
    # axs[2].set_yscale("log")
    axs[2].set_xlabel("sigma")
    axs[2].set_ylabel("value")
    axs[2].set_xlim(1e-2, 1e2)
    axs[2].set_ylim(0, 1)

    #plt.show()
    plt.tight_layout()
    plt.savefig("power_zs"+str(zs_val)+"_20221219.png",format='png')
    # print([power(zs_vals[0],s,data['sigma'],data['residues'],data['poles'],10000) for s in sigma_list])
    #power(zs_vals[0], 0.1, 0.1, data['sigma'], data['residues'], data['poles'], 10000)

if __name__ == '__main__':
    main()
