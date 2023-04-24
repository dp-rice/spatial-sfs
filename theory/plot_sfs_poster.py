import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import sys

sys.path.insert(1, 'numerics_snakemake')
from plot_sfs import *


def main():
    font = {'size': 18}

    plt.rc('font', **font)

    dim=2
    N=1000
    ymin = 1e-5
    ymax = 1e5
    sigma_vals_plot = [0.1,1,10,100]
    labs = [str(l) for l in sigma_vals_plot]
    s_vals = [0.001, 0.1]
    data = pd.read_csv("numerics_snakemake/cleaned_data_dim2_errorFalse.csv")  # always use errorFalse version to get higher orders
    data = data.loc[data['poly_type'] == '1_1']
    x_range = np.logspace(-6, -2, 100)
    fig, axs = plt.subplots(3, 2)
    axs[0,0].set_title("SFS for s=" + str(s_vals[0]))
    axs[0,0].set_xscale("log")
    axs[0,0].set_yscale("log")
    axs[0,0].set_ylim(ymin, ymax)
    colors = sns.color_palette("colorblind", len(sigma_vals_plot))
    for i in range(len(sigma_vals_plot)):
        axs[0,0].plot(x_range, gamma.pdf(x_range,
                                          a=shape_p(sigma=sigma_vals_plot[i], s=s_vals[0], sigma_vals=data['sigma'],
                                                    res_vals=data['residues'], d=dim, N=N),
                                          scale=1 / rate_p(sigma=sigma_vals_plot[i], s=s_vals[0],
                                                           sigma_vals=data['sigma'],
                                                           pole_vals=data['poles'], d=dim, N=N)),
                       color=colors[i],linewidth=24)
    axs[0, 0].set_xlabel("minor allele frequency")
    axs[0, 0].set_ylabel("density")
    # axs[0].legend(labels=labs, title="sigma")

    axs[0,1].set_title("SFS for s=" + str(s_vals[1]))
    axs[0,1].set_xscale("log")
    axs[0,1].set_yscale("log")
    axs[0,1].set_ylim(ymin, ymax)
    colors = sns.color_palette("colorblind", len(sigma_vals_plot))
    for i in range(len(sigma_vals_plot)):
        axs[0,1].plot(x_range, gamma.pdf(x_range,
                                          a=shape_p(sigma=sigma_vals_plot[i], s=s_vals[1], sigma_vals=data['sigma'],
                                                    res_vals=data['residues'], d=dim, N=N),
                                          scale=1 / rate_p(sigma=sigma_vals_plot[i], s=s_vals[1],
                                                           sigma_vals=data['sigma'],
                                                           pole_vals=data['poles'], d=dim, N=N)),
                       color=colors[i],linewidth=24)
    axs[0, 1].set_xlabel("minor allele frequency")
    axs[0, 1].set_ylabel("density")
    # axs[1].legend(labels=labs, title="sigma")

    # axs[2].set_title("s=" + str(s_vals[2]))
    # axs[2].set_xscale("log")
    # axs[2].set_yscale("log")
    # axs[2].set_ylim(ymin, ymax)
    # colors = sns.color_palette("colorblind", len(sigma_vals_plot))
    # for i in range(len(sigma_vals_plot)):
    #     axs[2].plot(x_range, gamma.pdf(x_range,
    #                                       a=shape_p(sigma=sigma_vals_plot[i], s=s_vals[2], sigma_vals=data['sigma'],
    #                                                 res_vals=data['residues'], d=dim, N=N),
    #                                       scale=1 / rate_p(sigma=sigma_vals_plot[i], s=s_vals[2],
    #                                                        sigma_vals=data['sigma'],
    #                                                        pole_vals=data['poles'], d=dim, N=N)),
    #                    color=colors[i],linewidth=24)
    axs[0,1].legend(labels=labs, title="sigma")

    colors = sns.color_palette("dark", 3)
    # fig, axs = plt.subplots(3, 3)
    sigma_range = np.logspace(-2, 2, 100)
    dim = 2
    N = 1000
    s_vals = [0.001, 0.01, 0.1]
    axs[1, 0].plot(sigma_range,
                   shape_p(s=s_vals[0], sigma=sigma_range, sigma_vals=data['sigma'].tolist(),
                           res_vals=data['residues'].tolist(),
                           d=dim, N=N), color=colors[0], linewidth=24)
    axs[1, 0].plot(sigma_range,
                   shape_p(s=s_vals[1], sigma=sigma_range, sigma_vals=data['sigma'].tolist(),
                           res_vals=data['residues'].tolist(),
                           d=dim, N=N), color=colors[1], linewidth=24)
    axs[1, 0].plot(sigma_range,
                   shape_p(s=s_vals[2], sigma=sigma_range, sigma_vals=data['sigma'].tolist(),
                           res_vals=data['residues'].tolis
    t(),
                           d=dim, N=N), color=colors[2], linewidth=24)
    # axs[0,0].legend(labels=['0.001', '0.01', '0.1'], title="s")
    axs[1, 0].set_xscale("log")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_title("Effective mutation supply")
    axs[1, 0].set_xlabel("sigma")
    axs[1, 0].set_ylabel("value")

    axs[1, 1].plot(sigma_range,
                   rate_p(s=s_vals[0], sigma=sigma_range, sigma_vals=data['sigma'].tolist(),
                          pole_vals=data['poles'].tolist(),
                          d=dim, N=N), color=colors[0], linewidth=24)
    axs[1, 1].plot(sigma_range,
                   rate_p(s=s_vals[1], sigma=sigma_range, sigma_vals=data['sigma'].tolist(),
                          pole_vals=data['poles'].tolist(),
                          d=dim, N=N), color=colors[1], linewidth=24)
    axs[1, 1].plot(sigma_range,
                   rate_p(s=s_vals[2], sigma=sigma_range, sigma_vals=data['sigma'].tolist(),
                          pole_vals=data['poles'].tolist(),
                          d=dim, N=N), color=colors[2], linewidth=24)
    axs[1, 1].legend(labels=['0.001', '0.01', '0.1'], title="s")
    axs[1, 1].set_xscale("log")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_title("Effective selection intensity")
    axs[1, 1].set_xlabel("sigma")
    axs[1,1].set_ylabel("value")

    colors = sns.color_palette("colorblind", 4)
    s_range = np.logspace(-6, 0, 100)
    axs[2, 0].plot(s_range,
                   shape_p(s=s_range, sigma=0.1, sigma_vals=data['sigma'].tolist(), res_vals=data['residues'].tolist(),
                           d=dim, N=N), color=colors[0], linewidth=24)
    axs[2, 0].plot(s_range,
                   shape_p(s=s_range, sigma=1, sigma_vals=data['sigma'].tolist(), res_vals=data['residues'].tolist(),
                           d=dim, N=N), color=colors[1], linewidth=24)
    axs[2, 0].plot(s_range,
                   shape_p(s=s_range, sigma=10, sigma_vals=data['sigma'].tolist(), res_vals=data['residues'].tolist(),
                           d=dim, N=N), color=colors[2], linewidth=24)
    axs[2, 0].plot(s_range,
                   shape_p(s=s_range, sigma=100, sigma_vals=data['sigma'].tolist(), res_vals=data['residues'].tolist(),
                           d=dim, N=N), color=colors[3], linewidth=24)
    # axs[1,0].legend(labels=['0.1','1', '10', '100'], title="sigma")
    axs[2, 0].set_xscale("log")
    axs[2, 0].set_yscale("log")
    axs[2, 0].set_title("Effective mutation supply")
    axs[2, 0].set_xlabel("s")
    axs[2, 0].set_ylabel("value")

    axs[2, 1].plot(s_range,
                   rate_p(s=s_range, sigma=0.1, sigma_vals=data['sigma'].tolist(), pole_vals=data['poles'].tolist(),
                          d=dim, N=N), color=colors[0], linewidth=24)
    axs[2, 1].plot(s_range,
                   rate_p(s=s_range, sigma=1, sigma_vals=data['sigma'].tolist(), pole_vals=data['poles'].tolist(),
                          d=dim, N=N), color=colors[1], linewidth=24)
    axs[2, 1].plot(s_range,
                   rate_p(s=s_range, sigma=10, sigma_vals=data['sigma'].tolist(), pole_vals=data['poles'].tolist(),
                          d=dim, N=N), color=colors[2], linewidth=24)
    axs[2, 1].plot(s_range,
                   rate_p(s=s_range, sigma=100, sigma_vals=data['sigma'].tolist(), pole_vals=data['poles'].tolist(),
                          d=dim, N=N), color=colors[3], linewidth=24)
    axs[2, 1].legend(labels=['0.1', '1', '10', '100'], title="sigma")
    axs[2, 1].set_xscale("log")
    axs[2, 1].set_yscale("log")
    axs[2, 1].set_title("Effective Selection Intensity")
    axs[2, 1].set_xlabel("s")
    axs[2, 1].set_ylabel("value")




    fig.set_figheight(12)
    fig.set_figwidth(12)
    plt.tight_layout()
    plt.savefig("sfs_w_params_for_poster_files_v2.pdf")

if __name__ == '__main__':
    main()
