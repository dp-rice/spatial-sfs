import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import sys
sys.path.insert(1, 'numerics_snakemake')
from plot_sfs import *


def exp_seg(sigma,s,sigma_vals,res_vals,pole_vals,n=1e6,Nval=10000):
    beta = rate_p(sigma,s,sigma_vals,pole_vals,N=Nval)
    alpha = shape_p(sigma,s,sigma_vals,res_vals,N=Nval)
    exp_val = 1-((1+(n/beta))**(-1*alpha))
    return(exp_val)

def sampling_effort(sigma,s,w,sigma_vals,res_vals,pole_vals,Nval=10000):
    beta = rate_p(sigma,s,sigma_vals,pole_vals,N=Nval)
    alpha = shape_p(sigma,s,sigma_vals,res_vals,N=Nval)
    effort_val = beta*((1-w)**(-1/alpha)-1)
    return(effort_val)


def main():
    # font = {'size': 18}
    #
    # plt.rc('font', **font)
    sigma_list = np.logspace(-2, 2, 100)
    data = pd.read_csv(
        "numerics_snakemake/cleaned_data_dim2_errorFalse.csv")  # always use errorFalse version to get higher orders
    data = data.loc[data['poly_type'] == '1_1']

    colors = sns.color_palette("dark", 3)
    fig,ax = plt.subplots()
    n_ax = 1e7
    s_vals = [0.001,0.01,0.1]
    ax.plot(sigma_list,
                  [exp_seg(sg, s_vals[0], data['sigma'], data['residues'], data['poles'], n=n_ax) for sg in sigma_list],
                  color=colors[0]),#linewidth=24)
    ax.plot(sigma_list,
                  [exp_seg(sg, s_vals[1], data['sigma'], data['residues'], data['poles'], n=n_ax) for sg in sigma_list],
                  color=colors[1]),#linewidth=24)
    ax.plot(sigma_list,
                  [exp_seg(sg, s_vals[2], data['sigma'], data['residues'], data['poles'], n=n_ax) for sg in sigma_list],
                  color=colors[2]),#linewidth=24)
    ax.legend(labels=[str(sv) for sv in s_vals],title="s")
    ax.set_title("Segregating sites")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("value")
    # plt.ylim(1e-3,1e-1)
    fig.set_figheight(3.622)
    fig.set_figwidth(5.748)
    plt.savefig("exp_seg_v2.pdf")

    fig,ax = plt.subplots()
    alpha_ref = shape_p(1e-2, s_vals[0], data['sigma'], data['residues'])
    stand_value = sampling_effort(1e-2, s_vals[0], alpha_ref / 100, data['sigma'], data['residues'], data['poles'])
    ax.plot(sigma_list, [
        sampling_effort(sg, s_vals[0], alpha_ref / 100, data['sigma'], data['residues'], data['poles']) / stand_value for
        sg in sigma_list],
               color=colors[0]),#linewidth=24)

    alpha_ref = shape_p(1e-2, s_vals[1], data['sigma'], data['residues'])
    stand_value = sampling_effort(1e-2, s_vals[1], alpha_ref / 100, data['sigma'], data['residues'], data['poles'])
    ax.plot(sigma_list, [
        sampling_effort(sg, s_vals[1], alpha_ref / 100, data['sigma'], data['residues'], data['poles']) / stand_value for
        sg in sigma_list],
               color=colors[1]),#linewidth=24)

    alpha_ref = shape_p(1e-2, s_vals[2], data['sigma'], data['residues'])
    stand_value = sampling_effort(1e-2, s_vals[2], alpha_ref / 100, data['sigma'], data['residues'], data['poles'])
    ax.plot(sigma_list, [
        sampling_effort(sg, s_vals[2], alpha_ref / 100, data['sigma'], data['residues'], data['poles']) / stand_value for
        sg in sigma_list],
               color=colors[2]),#linewidth=24)

    ax.set_title("Sampling effort")
    ax.set_xscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("value")
    # ax.set_ylim(1.0, 1.6)
    ax.legend(labels=[str(sv) for sv in s_vals], title="s")
    fig.set_figheight(3.622)
    fig.set_figwidth(5.748)
    plt.savefig("samp_effort_v2.pdf")


    # plot over N

    # exp seg
    N_vals = [100, 1000, 10000]
    fig, axs = plt.subplots(1, 3)

    # left
    axs[0].plot(sigma_list,
            [exp_seg(sg, s_vals[0], data['sigma'], data['residues'], data['poles'], n=n_ax,Nval=N_vals[0]) for sg in sigma_list],
            color=colors[0])
    axs[0].plot(sigma_list,
            [exp_seg(sg, s_vals[1], data['sigma'], data['residues'], data['poles'], n=n_ax,Nval=N_vals[0]) for sg in sigma_list],
            color=colors[1])
    axs[0].plot(sigma_list,
            [exp_seg(sg, s_vals[2], data['sigma'], data['residues'], data['poles'], n=n_ax,Nval=N_vals[0]) for sg in sigma_list],
            color=colors[2])
    axs[0].legend(labels=[str(sv) for sv in s_vals], title="s")
    axs[0].set_title("N="+str(N_vals[0]))
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("sigma")
    axs[0].set_ylabel("value")
    axs[0].set_xlim(1e-2, 1e2)
    axs[0].set_ylim(1e-5, 1)

    # middle
    axs[1].plot(sigma_list,
                   [exp_seg(sg, s_vals[0], data['sigma'], data['residues'], data['poles'], n=n_ax, Nval=N_vals[1]) for
                    sg in sigma_list],
                   color=colors[0])
    axs[1].plot(sigma_list,
                   [exp_seg(sg, s_vals[1], data['sigma'], data['residues'], data['poles'], n=n_ax, Nval=N_vals[1]) for
                    sg in sigma_list],
                   color=colors[1])
    axs[1].plot(sigma_list,
                   [exp_seg(sg, s_vals[2], data['sigma'], data['residues'], data['poles'], n=n_ax, Nval=N_vals[1]) for
                    sg in sigma_list],
                   color=colors[2])
    axs[1].legend(labels=[str(sv) for sv in s_vals], title="s")
    axs[1].set_title("N="+str(N_vals[1]))
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("sigma")
    axs[1].set_ylabel("value")
    axs[1].set_xlim(1e-2, 1e2)
    axs[1].set_ylim(1e-5, 1)
    # right
    axs[2].plot(sigma_list,
                   [exp_seg(sg, s_vals[0], data['sigma'], data['residues'], data['poles'], n=n_ax, Nval=N_vals[2]) for
                    sg in sigma_list],
                   color=colors[0])
    axs[2].plot(sigma_list,
                   [exp_seg(sg, s_vals[1], data['sigma'], data['residues'], data['poles'], n=n_ax, Nval=N_vals[2]) for
                    sg in sigma_list],
                   color=colors[1])
    axs[2].plot(sigma_list,
                   [exp_seg(sg, s_vals[2], data['sigma'], data['residues'], data['poles'], n=n_ax, Nval=N_vals[2]) for
                    sg in sigma_list],
                   color=colors[2])
    axs[2].legend(labels=[str(sv) for sv in s_vals], title="s")
    axs[2].set_title("N="+str(N_vals[2]))
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("sigma")
    axs[2].set_ylabel("value")
    axs[2].set_xlim(1e-2,1e2)
    axs[2].set_ylim(1e-5,1)
    plt.tight_layout()
    #plt.savefig("20221113_exp_seg_overN.pdf",bbox_inches="tight")

    # effort
    fig, axs = plt.subplots(1, 3)
    
    # left
    alpha_ref = shape_p(1e-2, s_vals[0], data['sigma'], data['residues'],N=N_vals[0])
    stand_value = sampling_effort(1e-2, s_vals[0], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],Nval=N_vals[0])
    axs[0].plot(sigma_list, [
        sampling_effort(sg, s_vals[0], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],Nval=N_vals[0]) / stand_value
        for
        sg in sigma_list],
            color=colors[0]),#linewidth=24)
    alpha_ref = shape_p(1e-2, s_vals[1], data['sigma'], data['residues'],N=N_vals[0])
    stand_value = sampling_effort(1e-2, s_vals[1], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],Nval=N_vals[0])
    axs[0].plot(sigma_list, [
        sampling_effort(sg, s_vals[1], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],Nval=N_vals[0]) / stand_value
        for
        sg in sigma_list],
            color=colors[1]),#linewidth=24)
    alpha_ref = shape_p(1e-2, s_vals[2], data['sigma'], data['residues'],N=N_vals[0])
    stand_value = sampling_effort(1e-2, s_vals[2], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],Nval=N_vals[0])
    axs[0].plot(sigma_list, [
        sampling_effort(sg, s_vals[2], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],Nval=N_vals[0]) / stand_value
        for
        sg in sigma_list],
            color=colors[2]),#linewidth=24)
    axs[0].set_title("N="+str(N_vals[0]))
    axs[0].set_xscale("log")
    axs[0].set_xlabel("sigma")
    axs[0].set_ylabel("value")
    # axs[0].set_ylim(1.0, 1.6)
    axs[0].legend(labels=[str(sv) for sv in s_vals], title="s")
    axs[0].set_xlim(1e-2, 1e2)
    axs[0].set_ylim(0.9, 2)

    # middle
    alpha_ref = shape_p(1e-2, s_vals[0], data['sigma'], data['residues'],N=N_vals[1])
    stand_value = sampling_effort(1e-2, s_vals[0], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],Nval=N_vals[1])
    axs[1].plot(sigma_list, [
        sampling_effort(sg, s_vals[0], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],
                        Nval=N_vals[1]) / stand_value
        for
        sg in sigma_list],
                color=colors[0]),#linewidth=24)
    alpha_ref = shape_p(1e-2, s_vals[1], data['sigma'], data['residues'],N=N_vals[1])
    stand_value = sampling_effort(1e-2, s_vals[1], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],Nval=N_vals[1])
    axs[1].plot(sigma_list, [
        sampling_effort(sg, s_vals[1], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],
                        Nval=N_vals[1]) / stand_value
        for
        sg in sigma_list],
                color=colors[1]),#linewidth=24)
    alpha_ref = shape_p(1e-2, s_vals[2], data['sigma'], data['residues'],N=N_vals[1])
    stand_value = sampling_effort(1e-2, s_vals[2], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],Nval=N_vals[1])
    axs[1].plot(sigma_list, [
        sampling_effort(sg, s_vals[2], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],
                        Nval=N_vals[1]) / stand_value
        for
        sg in sigma_list],
                color=colors[2]),#linewidth=24)
    axs[1].set_title("N=" + str(N_vals[1]))
    axs[1].set_xscale("log")
    axs[1].set_xlabel("sigma")
    axs[1].set_ylabel("value")
    # axs[0].set_ylim(1.0, 1.6)
    axs[1].legend(labels=[str(sv) for sv in s_vals], title="s")
    axs[1].set_xlim(1e-2, 1e2)
    axs[1].set_ylim(0.9, 2)

    # right
    alpha_ref = shape_p(1e-2, s_vals[0], data['sigma'], data['residues'],N=N_vals[2])
    print(alpha_ref) / 100
    stand_value = sampling_effort(1e-2, s_vals[0], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],Nval=N_vals[2])
    axs[2].plot(sigma_list, [
        sampling_effort(sg, s_vals[0], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],
                        Nval=N_vals[2]) / stand_value
        for
        sg in sigma_list],
                color=colors[0]),#linewidth=24)
    alpha_ref = shape_p(1e-2, s_vals[1], data['sigma'], data['residues'],N=N_vals[2])
    print(alpha_ref) / 100
    stand_value = sampling_effort(1e-2, s_vals[1], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],Nval=N_vals[2])
    axs[2].plot(sigma_list, [
        sampling_effort(sg, s_vals[1], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],
                        Nval=N_vals[2]) / stand_value
        for
        sg in sigma_list],
                color=colors[1]),#linewidth=24)
    alpha_ref = shape_p(1e-2, s_vals[2], data['sigma'], data['residues'],N=N_vals[2])
    print(alpha_ref)/100
    stand_value = sampling_effort(1e-2, s_vals[2], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],Nval=N_vals[2])
    axs[2].plot(sigma_list, [
        sampling_effort(sg, s_vals[2], alpha_ref / 100, data['sigma'], data['residues'], data['poles'],
                        Nval=N_vals[2]) / stand_value
        for
        sg in sigma_list],
                color=colors[2]),#linewidth=24)
    axs[2].set_title("N=" + str(N_vals[2]))
    axs[2].set_xscale("log")
    axs[2].set_xlabel("sigma")
    axs[2].set_ylabel("value")
    # axs[0].set_ylim(1.0, 1.6)
    axs[2].legend(labels=[str(sv) for sv in s_vals], title="s")
    axs[2].set_xlim(1e-2, 1e2)
    axs[2].set_ylim(0.9, 2)
    plt.tight_layout()
    plt.savefig("20221206_samp_effort_overN.pdf", bbox_inches="tight")



    # fig, ax = plt.subplots(2,3)
    #
    # # top left
    # n_ax = 1e3
    # ax[0,0].plot(sigma_list,[exp_seg(sg,1,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[0])
    # ax[0,0].plot(sigma_list,[exp_seg(sg,0.1,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[1])
    # ax[0,0].plot(sigma_list,[exp_seg(sg,0.05,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[2])
    # ax[0,0].plot(sigma_list,[exp_seg(sg,0.01,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[3])
    # # ax[0,0].legend(labels=['1','0.1','0.05','0.01'],title="s")
    # ax[0,0].set_title("n="+str(n_ax))
    # ax[0,0].set_xscale("log")
    # ax[0,0].set_yscale("log")
    #
    # # top middle
    # n_ax = 1e4
    # ax[0,1].plot(sigma_list,[exp_seg(sg,1,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[0])
    # ax[0,1].plot(sigma_list,[exp_seg(sg,0.1,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[1])
    # ax[0,1].plot(sigma_list,[exp_seg(sg,0.05,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[2])
    # ax[0,1].plot(sigma_list,[exp_seg(sg,0.01,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[3])
    # # ax[0,1].legend(labels=['1','0.1','0.05','0.01'],title="s")
    # ax[0,1].set_title("n="+str(n_ax))
    # ax[0,1].set_xscale("log")
    # ax[0,1].set_yscale("log")
    #
    # # top right
    # n_ax = 1e5
    # ax[0,2].plot(sigma_list,[exp_seg(sg,1,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[0])
    # ax[0,2].plot(sigma_list,[exp_seg(sg,0.1,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[1])
    # ax[0,2].plot(sigma_list,[exp_seg(sg,0.05,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[2])
    # ax[0,2].plot(sigma_list,[exp_seg(sg,0.01,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[3])
    # # ax[0,2].legend(labels=['1','0.1','0.05','0.01'],title="s")
    # ax[0,2].set_title("n="+str(n_ax))
    # ax[0,2].set_xscale("log")
    # ax[0,2].set_yscale("log")
    #
    # # bottom left
    # n_ax = 1e6
    # ax[1,0].plot(sigma_list,[exp_seg(sg,1,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[0])
    # ax[1,0].plot(sigma_list,[exp_seg(sg,0.1,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[1])
    # ax[1,0].plot(sigma_list,[exp_seg(sg,0.05,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[2])
    # ax[1,0].plot(sigma_list,[exp_seg(sg,0.01,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[3])
    # # ax[1,0].legend(labels=['1','0.1','0.05','0.01'],title="s")
    # ax[1,0].set_title("n="+str(n_ax))
    # ax[1,0].set_xscale("log")
    # ax[1,0].set_yscale("log")
    #
    # # bottom middle
    # n_ax = 1e7
    # ax[1,1].plot(sigma_list,[exp_seg(sg,1,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[0])
    # ax[1,1].plot(sigma_list,[exp_seg(sg,0.1,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[1])
    # ax[1,1].plot(sigma_list,[exp_seg(sg,0.05,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[2])
    # ax[1,1].plot(sigma_list,[exp_seg(sg,0.01,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[3])
    # # ax[1,1].legend(labels=['1','0.1','0.05','0.01'],title="s")
    # ax[1,1].set_title("n="+str(n_ax))
    # ax[1,1].set_xscale("log")
    # ax[1,1].set_yscale("log")
    #
    # # bottom right
    # n_ax = 1e8
    # ax[1,2].plot(sigma_list,[exp_seg(sg,1,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[0])
    # ax[1,2].plot(sigma_list,[exp_seg(sg,0.1,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[1])
    # ax[1,2].plot(sigma_list,[exp_seg(sg,0.05,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[2])
    # ax[1,2].plot(sigma_list,[exp_seg(sg,0.01,data['sigma'],data['residues'],data['poles'],n=n_ax) for sg in sigma_list],
    #             color=colors[3])
    # # ax[1,2].legend(labels=['1','0.1','0.05','0.01'],title="s")
    # ax[1,2].set_title("n="+str(n_ax))
    # ax[1,2].set_xscale("log")
    # ax[1,2].set_yscale("log")
    #
    # ax[0,0].set_ylim(1e-6,1e-1)
    # ax[0,1].set_ylim(1e-6,1e-1)
    # ax[0,2].set_ylim(1e-6,1e-1)
    # ax[1,0].set_ylim(1e-6,1e-1)
    # ax[1,1].set_ylim(1e-6,1e-1)
    # ax[1,2].set_ylim(1e-6,1e-1)
    #
    # plt.figlegend(labels=['1','0.1','0.05','0.01'],title="s")
    # # plt.show()



    # fig, ax = plt.subplots(1,3)
    #
    #
    # s_vals = 1
    # alpha_ref = shape_p(1e-2,s_vals,data['sigma'],data['residues'])
    # stand_value = sampling_effort(1e-2,s_vals,alpha_ref/10,data['sigma'],data['residues'],data['poles'])
    # ax[0].plot(sigma_list,[sampling_effort(sg,s_vals,alpha_ref/10,data['sigma'],data['residues'],data['poles'])/stand_value for sg in sigma_list],
    #             color=colors[0])
    #
    # s_vals = 0.1
    # alpha_ref = shape_p(1e-2,s_vals,data['sigma'],data['residues'])
    # stand_value = sampling_effort(1e-2,s_vals,alpha_ref/10,data['sigma'],data['residues'],data['poles'])
    # ax[0].plot(sigma_list,[sampling_effort(sg,s_vals,alpha_ref/10,data['sigma'],data['residues'],data['poles'])/stand_value for sg in sigma_list],
    #             color=colors[1])
    #
    # s_vals = 0.05
    # alpha_ref = shape_p(1e-2,s_vals,data['sigma'],data['residues'])
    # stand_value = sampling_effort(1e-2,s_vals,alpha_ref/10,data['sigma'],data['residues'],data['poles'])
    # ax[0].plot(sigma_list,[sampling_effort(sg,s_vals,alpha_ref/10,data['sigma'],data['residues'],data['poles'])/stand_value for sg in sigma_list],
    #             color=colors[2])
    #
    # s_vals = 0.01
    # alpha_ref = shape_p(1e-2,s_vals,data['sigma'],data['residues'])
    # stand_value = sampling_effort(1e-2,s_vals,alpha_ref/10,data['sigma'],data['residues'],data['poles'])
    # ax[0].plot(sigma_list,[sampling_effort(sg,s_vals,alpha_ref/10,data['sigma'],data['residues'],data['poles'])/stand_value for sg in sigma_list],
    #             color=colors[3])
    #
    #
    # ax[0].set_title("sampling effort alpha_ref/10")
    # ax[0].set_xscale("log")
    # ax[0].set_ylim(1.0, 1.6)
    # ax[0].legend(labels=['1','0.1','0.05','0.01'],title="s")
    # # plt.show()


    # fig, ax = plt.subplots(2)


    # s_vals = 1
    # alpha_ref = shape_p(1e-2,s_vals,data['sigma'],data['residues'])
    # stand_value = sampling_effort(1e-2,s_vals,alpha_ref/100,data['sigma'],data['residues'],data['poles'])
    # ax[1].plot(sigma_list,[sampling_effort(sg,s_vals,alpha_ref/100,data['sigma'],data['residues'],data['poles'])/stand_value for sg in sigma_list],
    #             color=colors[0])
    #
    # s_vals = 0.1
    # alpha_ref = shape_p(1e-2,s_vals,data['sigma'],data['residues'])
    # stand_value = sampling_effort(1e-2,s_vals,alpha_ref/100,data['sigma'],data['residues'],data['poles'])
    # ax[1].plot(sigma_list,[sampling_effort(sg,s_vals,alpha_ref/100,data['sigma'],data['residues'],data['poles'])/stand_value for sg in sigma_list],
    #             color=colors[1])
    #
    # s_vals = 0.05
    # alpha_ref = shape_p(1e-2,s_vals,data['sigma'],data['residues'])
    # stand_value = sampling_effort(1e-2,s_vals,alpha_ref/100,data['sigma'],data['residues'],data['poles'])
    # ax[1].plot(sigma_list,[sampling_effort(sg,s_vals,alpha_ref/100,data['sigma'],data['residues'],data['poles'])/stand_value for sg in sigma_list],
    #             color=colors[2])
    #
    # s_vals = 0.01
    # alpha_ref = shape_p(1e-2,s_vals,data['sigma'],data['residues'])
    # stand_value = sampling_effort(1e-2,s_vals,alpha_ref/100,data['sigma'],data['residues'],data['poles'])
    # ax[1].plot(sigma_list,[sampling_effort(sg,s_vals,alpha_ref/100,data['sigma'],data['residues'],data['poles'])/stand_value for sg in sigma_list],
    #             color=colors[3])
    #
    #
    # ax[1].set_title("sampling effort alpha_ref/100")
    # ax[1].set_xscale("log")
    # ax[1].set_ylim(1.0,1.6)
    # ax[1].legend(labels=['1','0.1','0.05','0.01'],title="s")
    #
    # s_vals = 1
    # alpha_ref = shape_p(1e-2, s_vals, data['sigma'], data['residues'])
    # stand_value = sampling_effort(1e-2, s_vals, alpha_ref / 1000, data['sigma'], data['residues'], data['poles'])
    # ax[2].plot(sigma_list, [
    #     sampling_effort(sg, s_vals, alpha_ref / 1000, data['sigma'], data['residues'], data['poles']) / stand_value for
    #     sg in sigma_list],
    #            color=colors[0])
    #
    # s_vals = 0.1
    # alpha_ref = shape_p(1e-2, s_vals, data['sigma'], data['residues'])
    # stand_value = sampling_effort(1e-2, s_vals, alpha_ref / 1000, data['sigma'], data['residues'], data['poles'])
    # ax[2].plot(sigma_list, [
    #     sampling_effort(sg, s_vals, alpha_ref / 1000, data['sigma'], data['residues'], data['poles']) / stand_value for
    #     sg in sigma_list],
    #            color=colors[1])
    #
    # s_vals = 0.05
    # alpha_ref = shape_p(1e-2, s_vals, data['sigma'], data['residues'])
    # stand_value = sampling_effort(1e-2, s_vals, alpha_ref / 1000, data['sigma'], data['residues'], data['poles'])
    # ax[2].plot(sigma_list, [
    #     sampling_effort(sg, s_vals, alpha_ref / 1000, data['sigma'], data['residues'], data['poles']) / stand_value for
    #     sg in sigma_list],
    #            color=colors[2])
    #
    # s_vals = 0.01
    # alpha_ref = shape_p(1e-2, s_vals, data['sigma'], data['residues'])
    # stand_value = sampling_effort(1e-2, s_vals, alpha_ref / 1000, data['sigma'], data['residues'], data['poles'])
    # ax[2].plot(sigma_list, [
    #     sampling_effort(sg, s_vals, alpha_ref / 1000, data['sigma'], data['residues'], data['poles']) / stand_value for
    #     sg in sigma_list],
    #            color=colors[3])
    #
    # ax[2].set_title("sampling effort alpha_ref/1000")
    # ax[2].set_xscale("log")
    # ax[2].set_ylim(1.0, 1.6)
    # ax[2].legend(labels=['1', '0.1', '0.05', '0.01'], title="s")
    #
    # plt.show()


if __name__ == '__main__':
    main()
