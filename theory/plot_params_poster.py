import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import sys

sys.path.insert(1, 'numerics_snakemake')
from plot_sfs import *

def main():
    font = {'size': 22}
    plt.rc('font', **font)

    data = pd.read_csv(
        "numerics_snakemake/cleaned_data_dim2_errorFalse.csv")  # always use errorFalse version to get higher orders
    data = data.loc[data['poly_type'] == '1_1']
    colors = sns.color_palette("dark", 3)
    fig, axs = plt.subplots(2, 2)
    sigma_range = np.logspace(-2,2,100)
    dim = 2
    N = 10000
    s_vals = [0.001, 0.01, 0.1]
    axs[0,0].plot(sigma_range,
                shape_p(s=s_vals[0], sigma=sigma_range, sigma_vals=data['sigma'].tolist(), res_vals=data['residues'].tolist(),
                        d=dim, N=N), color=colors[0])#,linewidth=32)
    axs[0,0].plot(sigma_range,
                shape_p(s=s_vals[1], sigma=sigma_range, sigma_vals=data['sigma'].tolist(), res_vals=data['residues'].tolist(),
                        d=dim, N=N), color=colors[1])#,linewidth=32)
    axs[0,0].plot(sigma_range,
                shape_p(s=s_vals[2], sigma=sigma_range, sigma_vals=data['sigma'].tolist(), res_vals=data['residues'].tolist(),
                        d=dim, N=N), color=colors[2])#,linewidth=32)
    # axs[0,0].legend(labels=['0.001', '0.01', '0.1'], title="s")
    axs[0,0].set_xscale("log")
    axs[0,0].set_yscale("log")
    axs[0,0].set_title("Effective mutation supply")
    axs[0,0].set_xlabel("sigma")
    axs[0, 0].set_ylim(1e-6,1)

    axs[0,1].plot(sigma_range,
                rate_p(s=s_vals[0], sigma=sigma_range, sigma_vals=data['sigma'].tolist(), pole_vals=data['poles'].tolist(),
                       d=dim, N=N), color=colors[0])#,linewidth=32)
    axs[0,1].plot(sigma_range,
                rate_p(s=s_vals[1], sigma=sigma_range, sigma_vals=data['sigma'].tolist(), pole_vals=data['poles'].tolist(),
                       d=dim, N=N), color=colors[1])#,linewidth=32)
    axs[0,1].plot(sigma_range,
                rate_p(s=s_vals[2], sigma=sigma_range, sigma_vals=data['sigma'].tolist(), pole_vals=data['poles'].tolist(),
                       d=dim, N=N), color=colors[2])#,linewidth=32)
    axs[0,1].legend(labels=['0.001', '0.01', '0.1'], title="s")
    axs[0,1].set_xscale("log")
    axs[0,1].set_yscale("log")
    axs[0,1].set_title("Effective selection intensity")
    axs[0,1].set_xlabel("sigma")
    axs[0, 1].set_ylim(1e2, 1e7)

    colors = sns.color_palette("colorblind", 4)
    s_range = np.logspace(-6, 0, 100)
    axs[1, 0].plot(s_range,
                   shape_p(s=s_range, sigma=0.1, sigma_vals=data['sigma'].tolist(), res_vals=data['residues'].tolist(),
                           d=dim, N=N), color=colors[0])#,linewidth=32)
    axs[1,0].plot(s_range,
                shape_p(s=s_range, sigma=1, sigma_vals=data['sigma'].tolist(), res_vals=data['residues'].tolist(),
                        d=dim, N=N), color=colors[1])#,linewidth=32)
    axs[1,0].plot(s_range,
                shape_p(s=s_range, sigma=10, sigma_vals=data['sigma'].tolist(), res_vals=data['residues'].tolist(),
                        d=dim, N=N), color=colors[2])#,linewidth=32)
    axs[1,0].plot(s_range,
                shape_p(s=s_range, sigma=100, sigma_vals=data['sigma'].tolist(), res_vals=data['residues'].tolist(),
                        d=dim, N=N), color=colors[3])#,linewidth=32)
    # axs[1,0].legend(labels=['0.1','1', '10', '100'], title="sigma")
    axs[1,0].set_xscale("log")
    axs[1,0].set_yscale("log")
    axs[1,0].set_title("Effective Mutation Supply")
    axs[1,0].set_xlabel("s")
    axs[1, 0].set_ylim(1e-5, 1e2)

    axs[1, 1].plot(s_range,
                   rate_p(s=s_range, sigma=0.1, sigma_vals=data['sigma'].tolist(), pole_vals=data['poles'].tolist(),
                          d=dim, N=N), color=colors[0])#)#,linewidth=32)
    axs[1,1].plot(s_range,
                rate_p(s=s_range, sigma=1, sigma_vals=data['sigma'].tolist(), pole_vals=data['poles'].tolist(),
                       d=dim, N=N), color=colors[1])#)#,linewidth=32)
    axs[1,1].plot(s_range,
                rate_p(s=s_range, sigma=10, sigma_vals=data['sigma'].tolist(), pole_vals=data['poles'].tolist(),
                       d=dim, N=N), color=colors[2])#)#,linewidth=32)
    axs[1,1].plot(s_range,
                rate_p(s=s_range, sigma=100, sigma_vals=data['sigma'].tolist(), pole_vals=data['poles'].tolist(),
                       d=dim, N=N), color=colors[3])#)#,linewidth=32)
    axs[1,1].legend(labels=['0.1','1', '10', '100'], title="sigma")
    axs[1,1].set_xscale("log")
    axs[1,1].set_yscale("log")
    axs[1,1].set_title("Effective selection intensity")
    axs[1,1].set_xlabel("s")
    axs[1, 1].set_ylim(1e2, 1e8)

    fig.set_figheight(12)
    fig.set_figwidth(12)
    plt.tight_layout()

    plt.savefig("20221113_plots_params_sigma_s_dim_" + str(dim) + "_polytype_" + '1_1' + "_N" + str(N) + ".pdf")

if __name__ == '__main__':
    main()