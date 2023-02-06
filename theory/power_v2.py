from scipy.special import erfinv
from scipy.stats import gamma
import sys
import math
sys.path.insert(1,'numerics_snakemake')
from plot_sfs import *
import matplotlib.cm as cm
import matplotlib as mpl


def power(zs, sigma, s, sigma_vals, res_vals, pole_vals, Nval=10000):
    beta = rate_p(sigma,s,sigma_vals,pole_vals,N=Nval)
    alpha = shape_p(sigma,s,sigma_vals,res_vals,N=Nval)
    #return(1-gamma.cdf(zs,a=alpha,scale=1/beta))
    return(gamma.sf(zs,a=alpha,scale=1/beta))

def calc_V_G(kappa_list,s,sel_grad=1.0,L=1000,mu=1e-8,d=2,D=1,N=10000):
    V_list = [(2*L*mu*k)/(sel_grad**2*np.sqrt(D/s)**d*N) for k in kappa_list]
    return(V_list)

def calc_V_E(sigma,sigma2R,sigma2G,l=1,d=2):
    coef=1-1/((1+2*(sigma/l)**2)**(d/2))
    return(sigma2R+coef*sigma2G)

def get_zs(s,sel_grad,Vp,n,a=1e-8):
    vs = 2*erfinv(1-a)**2
    # print((2*vs*(Vp/n))/((s*sel_grad)**2))
    return(0.5 - (0.5)*math.sqrt(1-(2*vs*(Vp/n))/((s*sel_grad)**2)))

def power_all(sigma,s,sigma_vals,kappa_list,
              res_vals,pole_vals,
              sigma2R=0.1,sigma2G=1.0,Nval=10000,n=1e6,a=1e-8,L=10000,mu=1e-8,d=2,D=1,l=1,sel_grad=1.0):
    Vg=calc_V_G(kappa_list,s,sel_grad,N=Nval,L=L,mu=mu,d=d,D=D)[sigma_vals.index(sigma)]
    Ve=calc_V_E(sigma,sigma2R,sigma2G,l,d)
    Vp=Vg+Ve
    # print(Vp)
    # print(s)
    zs=get_zs(s,sel_grad=sel_grad,Vp=Vp,n=n,a=a)
    # print(zs)
    power_val = power(zs,sigma,s,sigma_vals,res_vals,pole_vals,Nval)
    return(power_val)

def power_all_zs(sigma,s,sigma_vals,kappa_list,
              res_vals,pole_vals,
              sigma2R=0.1,sigma2G=1.0,Nval=10000,n=1e6,a=1e-8,L=10000,mu=1e-8,d=2,D=1,l=1,sel_grad=1.0):
    Vg=calc_V_G(kappa_list,s,sel_grad,N=Nval,L=L,mu=mu,d=d,D=D)[sigma_vals.index(sigma)]
    Ve=calc_V_E(sigma,sigma2R,sigma2G,l,d)
    Vp=Vg+Ve
    # print(Vp)
    # print(s)
    zs=get_zs(s,sel_grad=sel_grad,Vp=Vp,n=n,a=a)
    # print(zs)
    power_val = power(zs,sigma,s,sigma_vals,res_vals,pole_vals,Nval)
    return zs, power_val

def zs_all(sigma,s,sigma_vals,kappa_list,
              res_vals,pole_vals,
              sigma2R=0.1,sigma2G=1.0,Nval=10000,n=1e6,a=1e-8,L=10000,mu=1e-8,d=2,D=1,l=1,sel_grad=1.0):
    Vg=calc_V_G(kappa_list,s,sel_grad,N=Nval,L=L,mu=mu,d=d,D=D)[sigma_vals.index(sigma)]
    Ve=calc_V_E(sigma,sigma2R,sigma2G,l,d)
    Vp=Vg+Ve
    # print(Vp)
    # print(s)
    zs=get_zs(s,sel_grad=sel_grad,Vp=Vp,n=n,a=a)
    # print(zs)
    #power_val = power(zs,sigma,s,sigma_vals,res_vals,pole_vals,Nval)
    return(zs)

def main():
    data = pd.read_csv("numerics_snakemake/spatial_integrals_dim2.csv")
    data_pr = pd.read_csv("numerics_snakemake/cleaned_data_dim2_errorFalse.csv")
    data_pr = data_pr.loc[data_pr['poly_type'] == '1_1']

    sigma_vals = data['sigma'].tolist()
    kappa_list = data['u2_GQ'].tolist()
    res_vals = data_pr['residues']
    pole_vals = data_pr['poles']

    sg_point = 0.01

    n_list = np.logspace(1,8)
    fig, ax = plt.subplots(1,1)
    ax.plot(n_list,[power_all(sg_point,0.1,sigma_vals,kappa_list,res_vals,pole_vals,sigma2G=0,sigma2R=1e-8,n=nv) for nv in n_list])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample size (n)")
    ax.set_ylabel("power")
    plt.tight_layout()
    plt.savefig("samplesize_power_point.png",format='png')

    ve_list = np.logspace(-8, -1)
    fig, ax = plt.subplots(1, 1)
    ax.plot(ve_list,
            [power_all(sg_point, 0.1, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=0, sigma2R=ve) for ve
             in ve_list])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("V_E")
    ax.set_ylabel("power")
    plt.savefig("envvar_power_point.png", format='png')

    sigma2G_list = np.logspace(-8, -2)
    fig, ax = plt.subplots(1, 1)
    ax.plot(sigma2G_list,
            [power_all(0.01, 0.1, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=s2g, sigma2R=1e-8) for s2g
             in sigma2G_list])
    ax.plot(sigma2G_list,
            [power_all(sigma_vals[20], 0.1, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=s2g, sigma2R=1e-8) for s2g
             in sigma2G_list])
    ax.plot(sigma2G_list,
            [power_all(sigma_vals[50], 0.1, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=s2g, sigma2R=1e-8) for
             s2g
             in sigma2G_list])
    ax.plot(sigma2G_list,
            [power_all(sigma_vals[75], 0.1, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=s2g, sigma2R=1e-8) for s2g
             in sigma2G_list])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sigma_G^2")
    ax.set_ylabel("power")
    plt.legend(["point (sigma=0.01)","narrow (sigma="+str(round(sigma_vals[20],2))+")",'broad (sigma='+str(round(sigma_vals[50],2))+")",'broad (sigma='+str(round(sigma_vals[75],2))+")"])
    # plt.show()
    plt.savefig("power_over_sigma2g.png", format='png')

    sigma2G_list = np.logspace(-8, -2)
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]})
    colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_vals)))
    norm = mpl.colors.LogNorm(vmin=min(sigma_vals), vmax=max(sigma_vals))
    for i in range(len(sigma_vals)):
        axs[0].plot(sigma2G_list,
                [power_all(sigma_vals[i], 0.1, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=s2g, sigma2R=1e-8) for s2g
                 in sigma2G_list],color=colors[i])
    # ax.plot(sigma2G_list,
    #         [power_all(0.01, 0.1, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=s2g, sigma2R=1e-8) for s2g
    #          in sigma2G_list])
    # ax.plot(sigma2G_list,
    #         [power_all(sigma_vals[20], 0.1, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=s2g, sigma2R=1e-8) for
    #          s2g
    #          in sigma2G_list])
    # ax.plot(sigma2G_list,
    #         [power_all(sigma_vals[50], 0.1, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=s2g, sigma2R=1e-8) for
    #          s2g
    #          in sigma2G_list])
    # ax.plot(sigma2G_list,
    #         [power_all(sigma_vals[75], 0.1, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=s2g, sigma2R=1e-8) for
    #          s2g
    #          in sigma2G_list])
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("sigma_G^2")
    axs[0].set_ylabel("power")
    # plt.legend(["point (sigma=0.01)", "narrow (sigma=" + str(round(sigma_vals[20], 2)) + ")",
    #             'broad (sigma=' + str(round(sigma_vals[50], 2)) + ")",
    #             'broad (sigma=' + str(round(sigma_vals[75], 2)) + ")"])
    # plt.show()
    cb1 = mpl.colorbar.ColorbarBase(axs[1], cmap=mpl.cm.viridis,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('sigma')
    plt.savefig("power_over_sigma2g_v2.png", format='png')

    sigma2G_list = np.logspace(-8, -2,30)
    fig, axs = plt.subplots(1, 2,gridspec_kw={'width_ratios': [10, 1]})
    colors = plt.cm.viridis(np.linspace(0, 1, 30))
    norm = mpl.colors.LogNorm(vmin=min(sigma2G_list), vmax=max(sigma2G_list))
    for i in range(30):
        axs[0].plot(sigma_vals,[power_all(sg, 0.01, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=sigma2G_list[i],
                          sigma2R=0)
                for sg in sigma_vals],color=colors[i])

    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("sigma")
    axs[0].set_ylabel("power")
    # plt.legend(["point (sigma=0.01)", "narrow (sigma=" + str(round(sigma_vals[20], 2)) + ")",
    #             'broad (sigma=' + str(round(sigma_vals[50], 2)) + ")",
    #             'broad (sigma=' + str(round(sigma_vals[75], 2)) + ")"])
    # plt.show()
    cb1 = mpl.colorbar.ColorbarBase(axs[1], cmap=mpl.cm.viridis,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('sigma_G^2')
    plt.savefig("power_over_sigma2g_sigma_v2.png", format='png')


    s_list = [0.001, 0.01, 0.1]
    sigma2G_list = [0, 1e-4, 1e-2]
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            axs[i,j].plot(sigma_vals, [
                power_all(sg, s_list[i], sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=sigma2G_list[j], sigma2R=1e-8)
                for sg in sigma_vals])
            axs[i,j].set_xscale("log")
            axs[i,j].set_yscale("log")
            axs[i,j].set_title("s=" + str(s_list[i])+", sigma2G=" + str(sigma2G_list[j]))
            # axs[i].set_ylim(1e-15, 1e-6)
            axs[i,j].set_xlabel("sigma")
            axs[i,j].set_ylabel("Power")
    plt.tight_layout()
    plt.savefig("power_over_sigma_s_sigma2G.png", format='png')

    s_list = [0.001, 0.01, 0.1]
    sigma2G_list = [0, 1e-4, 1e-2]
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            axs[i, j].plot(sigma_vals, [
                power_all(sg, s_list[i], sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=sigma2G_list[j],
                          sigma2R=1e-8)
                for sg in sigma_vals])
            axs[i, j].set_xscale("log")
            axs[i, j].set_yscale("log")
            axs[i, j].set_title("s=" + str(s_list[i]) + ", sigma2G=" + str(sigma2G_list[j]))
            # axs[i].set_ylim(1e-15, 1e-6)
            axs[i, j].set_xlabel("sigma")
            axs[i, j].set_ylabel("Power")
            axs[i,j].set_ylim(1e-15,1)
    plt.tight_layout()
    plt.savefig("power_over_sigma_s_sigma2G_v2.png", format='png')

    s_list = [0.001, 0.01, 0.1]
    n_list = [1e4,1e5,1e6]
    # sigma2G_list = [0, 1e-2, 1e-4]
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            axs[i, j].plot(sigma_vals, [
                power_all(sg, s_list[i], sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=1e-4,
                          sigma2R=1e-8,n=n_list[j])
                for sg in sigma_vals])
            axs[i, j].set_xscale("log")
            axs[i, j].set_yscale("log")
            axs[i, j].set_title("s=" + str(s_list[i]) + ", n=" + str(n_list[j]))
            # axs[i].set_ylim(1e-15, 1e-6)
            axs[i, j].set_xlabel("sigma")
            axs[i, j].set_ylabel("Power")
    plt.tight_layout()
    plt.savefig("power_over_sigma_s_n.png", format='png')


      # plot Vg over sigma
    mu_list = [1e-8, 1e-6, 1e-4]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axs[i].plot(sigma_vals, calc_V_G(kappa_list, s=0.01, mu=mu_list[i]))
        axs[i].set_xscale("log")
        axs[i].set_yscale("log")
        axs[i].set_title("s=0.001, mutation rate " + str(mu_list[i]))
        axs[i].set_ylim(1e-15, 1e-6)
        axs[i].set_xlabel("sigma")
        axs[i].set_ylabel("V_G")
    plt.tight_layout()
    plt.savefig("plots_20230125/VG_sigma_mu.png",format='png')

    # plot Ve over sigma
    sigma2G_list = [0,0.01, 1]#[0.1, 1, 10]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axs[i].plot(sigma_vals, [calc_V_E(sg,sigma2R=0.1,sigma2G=sigma2G_list[i]) for sg in sigma_vals])
        axs[i].set_xscale("log")
        axs[i].set_yscale("log")
        axs[i].set_title("sigma2R=0.1, sigma2G=" + str(sigma2G_list[i]))
        axs[i].set_ylim(1e-1, 10)
        axs[i].set_xlabel("sigma")
        axs[i].set_ylabel("V_E")
    plt.tight_layout()
    plt.savefig("plots_20230125/VE_sigma_sigma2G.png",format='png')
    #
    # plot power over sigma

    # def power_all(sigma, s, sigma_vals, kappa_list,
    #               res_vals, pole_vals,
    #               sigma2R=0.1, sigma2G=1, Nval=10000, n=10000, a=1e-8, L=10000, mu=1e-8, d=2, D=1, l=1):
    #     Vg = calc_V_G(kappa_list, s, sel_grad=1.0, N=Nval, L=L, mu=mu, d=d, D=D)[sigma_vals.index(sigma)]
    #     Ve = calc_V_E(sigma, sigma2R, sigma2G, l, d)
    #     Vp = Vg + Ve
    #     zs = get_zs(s, sel_grad=sel_grad, Vp=Vp, n=n, a=a)
    #     power_val = power(zs, sigma, s, sigma_vals, res_vals, pole_vals, Nval)
    #     return (power_val)

    # s_list = [1,1,1]
    # sigma2G_list = [0,0.01, 1]
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(3):
    #     axs[i].plot(sigma_vals, [power_all(sg,s_list[i],sigma_vals,kappa_list,res_vals,pole_vals,sigma2G=sigma2G_list[i],sigma2R=0.1) for sg in sigma_vals])
    #     axs[i].set_xscale("log")
    #     axs[i].set_yscale("log")
    #     axs[i].set_title("s="+ str(s_list[i])+", sigma2G="+str(sigma2G_list[i])+", sigma2R=0.1")
    #     # axs[i].set_ylim(1e-15, 1e-6)
    #     axs[i].set_xlabel("sigma")
    #     axs[i].set_ylabel("Power")
    # plt.tight_layout()
    # plt.savefig("plots_20230125/power_sigma_s1_sigma2R01.png",format='png')
    #
    # s_list = [0.01,0.01,0.01]
    # sigma2G_list = [0,0.01, 1]
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(3):
    #     axs[i].plot(sigma_vals, [power_all(sg,s_list[i],sigma_vals,kappa_list,res_vals,pole_vals,sigma2G=sigma2G_list[i],sigma2R=0.1) for sg in sigma_vals])
    #     # axs[i].set_xscale("log")
    #     # axs[i].set_yscale("log")
    #     axs[i].set_title("s="+ str(s_list[i])+", sigma2G="+str(sigma2G_list[i])+", sigma2R=0.1")
    #     # axs[i].set_ylim(1e-15, 1e-6)
    #     axs[i].set_xlabel("sigma")
    #     axs[i].set_ylabel("Power")
    # plt.tight_layout()
    # plt.savefig("plots_20230125/power_sigma_s01_sigma2R01.png",format='png')
    #
    # s_list = [1, 1, 1]
    # sigma2G_list = [0, 0.01, 1]
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(3):
    #     axs[i].plot(sigma_vals, [
    #         power_all(sg, s_list[i], sigma_vals, kappa_list, res_vals, pole_vals,sigma2G=sigma2G_list[i],
    #                   sigma2R=1e-8) for sg in sigma_vals])
    #     axs[i].set_xscale("log")
    #     axs[i].set_yscale("log")
    #     axs[i].set_title("s=" + str(s_list[i]) + ", sigma2G=" + str(sigma2G_list[i]) + ", sigma2R=1e-8")
    #     # axs[i].set_ylim(1e-15, 1e-6)
    #     axs[i].set_xlabel("sigma")
    #     axs[i].set_ylabel("Power")
    # plt.tight_layout()
    # plt.savefig("plots_20230125/power_sigma_s1_sigma2R1e-8.png",format='png')
    #
    # s_list = [0.01, 0.01, 0.01]
    # sigma2G_list = [0, 0.01, 1]
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(3):
    #     axs[i].plot(sigma_vals, [
    #         power_all(sg, s_list[i], sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=sigma2G_list[i],
    #                   sigma2R=1e-8) for sg in sigma_vals])
    #     axs[i].set_xscale("log")
    #     axs[i].set_yscale("log")
    #     axs[i].set_title("s=" + str(s_list[i]) + ", sigma2G=" + str(sigma2G_list[i]) + ", sigma2R=1e-8")
    #     # axs[i].set_ylim(1e-15, 1e-6)
    #     axs[i].set_xlabel("sigma")
    #     axs[i].set_ylabel("Power")
    # plt.tight_layout()
    # plt.savefig("plots_20230125/power_sigma_s01_sigma2R1e-8.png",format='png')
    #
    # # s_list = [2,2,2]
    # # sigma2G_list = [0, 0.01, 1]
    # # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # # for i in range(3):
    # #     axs[i].plot(sigma_vals, [
    # #         power_all(sg, s_list[i], sigma_vals, kappa_list, res_vals, pole_vals, sigma2R=1e-8, sigma2G=sigma2G_list[i])
    # #         for sg in sigma_vals])
    # #     axs[i].set_xscale("log")
    # #     axs[i].set_yscale("log")
    # #     axs[i].set_title("s=" + str(s_list[i]) + ", sigma2G=" + str(sigma2G_list[i]))
    # #     # axs[i].set_ylim(1e-15, 1e-6)
    # #     axs[i].set_xlabel("sigma")
    # #     axs[i].set_ylabel("Power")
    # # plt.tight_layout()
    # # plt.show()
    #
    # s_list = [1, 1, 1]
    # sigma2G_list = [0, 0.01, 1]
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(3):
    #     axs[i].plot(sigma_vals, [
    #         power_all(sg, s_list[i], sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=sigma2G_list[i],
    #                   sigma2R=0) for sg in sigma_vals])
    #     axs[i].set_xscale("log")
    #     axs[i].set_yscale("log")
    #     axs[i].set_title("s=" + str(s_list[i]) + ", sigma2G=" + str(sigma2G_list[i]) + ", sigma2R=0")
    #     # axs[i].set_ylim(1e-15, 1e-6)
    #     axs[i].set_xlabel("sigma")
    #     axs[i].set_ylabel("Power")
    # plt.tight_layout()
    # plt.savefig("plots_20230125/power_sigma_s1_sigma2R0.png",format='png')
    #
    # s_list = [0.01, 0.01, 0.01]
    # sigma2G_list = [0, 0.01, 1]
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(3):
    #     axs[i].plot(sigma_vals, [
    #         power_all(sg, s_list[i], sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=sigma2G_list[i],
    #                   sigma2R=0) for sg in sigma_vals])
    #     axs[i].set_xscale("log")
    #     axs[i].set_yscale("log")
    #     axs[i].set_title("s=" + str(s_list[i]) + ", sigma2G=" + str(sigma2G_list[i]) + ", sigma2R=0")
    #     # axs[i].set_ylim(1e-15, 1e-6)
    #     axs[i].set_xlabel("sigma")
    #     axs[i].set_ylabel("Power")
    # plt.tight_layout()
    # plt.savefig("plots_20230125/power_sigma_s01_sigma2R0.png",format='png')
    #
    # s_list = [1, 1, 1]
    # sigma2G_list = [0.01,0.1,1]
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(3):
    #     axs[i].plot(sigma_vals, [
    #         calc_V_G(kappa_list,s_list[i])[sigma_vals.index(sg)]/(calc_V_G(kappa_list,s_list[i])[sigma_vals.index(sg)]+calc_V_E(sg,sigma2R=1e-8,sigma2G=sigma2G_list[i])) for sg in sigma_vals])
    #     axs[i].set_xscale("log")
    #     axs[i].set_yscale("log")
    #     axs[i].set_title("s=" + str(s_list[i]) + ", sigma2G=" + str(sigma2G_list[i]) + ", sigma2R=1e-8")
    #     # axs[i].set_ylim(1e-15, 1e-6)
    #     axs[i].set_xlabel("sigma")
    #     axs[i].set_ylabel("h^2")
    # plt.tight_layout()
    # plt.savefig("plots_20230125/heritability_v1.png", format='png')
    #
    # s_list = [0.01, 0.1, 1]
    # sigma2G_list = [1,1, 1]
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(3):
    #     axs[i].plot(sigma_vals, [
    #         calc_V_G(kappa_list, s_list[i])[sigma_vals.index(sg)] / (
    #                     calc_V_G(kappa_list, s_list[i])[sigma_vals.index(sg)] + calc_V_E(sg, sigma2R=1e-8,
    #                                                                                      sigma2G=sigma2G_list[i])) for
    #         sg in sigma_vals])
    #     axs[i].set_xscale("log")
    #     axs[i].set_yscale("log")
    #     axs[i].set_title("s=" + str(s_list[i]) + ", sigma2G=" + str(sigma2G_list[i]) + ", sigma2R=1e-8")
    #     # axs[i].set_ylim(1e-15, 1e-6)
    #     axs[i].set_xlabel("sigma")
    #     axs[i].set_ylabel("h^2")
    # plt.tight_layout()
    # plt.savefig("plots_20230125/heritability_v2.png", format='png')
    #
    s_list = [0.001, 0.01, 0.1]
    sigma2G_list = [0,1e-4,1e-2]
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            axs[i,j].plot(sigma_vals, [
                calc_V_G(kappa_list, s_list[i])[sigma_vals.index(sg)] / (
                            calc_V_G(kappa_list, s_list[i])[sigma_vals.index(sg)] + calc_V_E(sg, sigma2R=0,
                                                                                             sigma2G=sigma2G_list[j])) for
                sg in sigma_vals])
            axs[i,j].set_xscale("log")
            axs[i,j].set_yscale("log")
            axs[i,j].set_title("s=" + str(s_list[i]) + ", sigma2G=" + str(sigma2G_list[j]) + ", sigma2R=0")
            # axs[i].set_ylim(1e-15, 1e-6)
            axs[i,j].set_xlabel("sigma")
            axs[i,j].set_ylabel("h^2")
    plt.tight_layout()
    plt.savefig("heritability_s_sigma2G.png", format='png')

    #   # plot Vg over sigma
    # mu_list = [1e-8, 1e-6, 1e-4]
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(3):
    #     axs[i].plot(sigma_vals, calc_V_G(kappa_list, s=0.01, mu=mu_list[i]))
    #     axs[i].set_xscale("log")
    #     axs[i].set_yscale("log")
    #     axs[i].set_title("s=0.001, mutation rate " + str(mu_list[i]))
    #     axs[i].set_ylim(1e-15, 1e-6)
    #     axs[i].set_xlabel("sigma")
    #     axs[i].set_ylabel("V_G")
    # plt.tight_layout()
    # plt.savefig("plots_20230125/VG_sigma_mu.png",format='png')
    #
    # # plot Ve over sigma
    # sigma2G_list = [0,0.01, 1]#[0.1, 1, 10]
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(3):
    #     axs[i].plot(sigma_vals, [calc_V_E(sg,sigma2R=0.1,sigma2G=sigma2G_list[i]) for sg in sigma_vals])
    #     axs[i].set_xscale("log")
    #     axs[i].set_yscale("log")
    #     axs[i].set_title("sigma2R=0.1, sigma2G=" + str(sigma2G_list[i]))
    #     axs[i].set_ylim(1e-1, 10)
    #     axs[i].set_xlabel("sigma")
    #     axs[i].set_ylabel("V_E")
    # plt.tight_layout()
    # plt.savefig("plots_20230125/VE_sigma_sigma2G.png",format='png')

    # fig,ax=plt.subplots(1,1)
    # ax.plot(sigma_vals,calc_V_G(kappa_list, s=0.01, mu=1e-8))
    # ax.plot(sigma_vals, [calc_V_E(sg,sigma2R=1e-8,sigma2G=0.01) for sg in sigma_vals])
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # plt.show()

    sigma2G_list = [0,0.01, 1]#[0.1, 1, 10]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axs[i].plot(sigma_vals, [calc_V_E(sg,sigma2R=1e-12,sigma2G=sigma2G_list[i]) for sg in sigma_vals])
        axs[i].plot(sigma_vals, calc_V_G(kappa_list, s=0.01, mu=1e-8))
        axs[i].set_xscale("log")
        axs[i].set_yscale("log")
        axs[i].set_title("sigma2R=1e-12, sigma2G=" + str(sigma2G_list[i]))
        # axs[i].set_ylim(1e-1, 10)
        axs[i].set_xlabel("sigma")
        axs[i].set_ylabel("value")
    plt.tight_layout()
    plt.savefig("plots_20230206/VE_vs_VG.png",format='png')

    sigma2G_list = np.logspace(-8, -2, 30)
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]})
    colors = plt.cm.viridis(np.linspace(0, 1, 30))
    norm = mpl.colors.LogNorm(vmin=min(sigma2G_list), vmax=max(sigma2G_list))
    for i in range(30):
        axs[0].plot(sigma_vals,
                    [zs_all(sg, 0.01, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=sigma2G_list[i],
                               sigma2R=0)
                     for sg in sigma_vals], color=colors[i])

    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("sigma")
    axs[0].set_ylabel("zs")
    # plt.legend(["point (sigma=0.01)", "narrow (sigma=" + str(round(sigma_vals[20], 2)) + ")",
    #             'broad (sigma=' + str(round(sigma_vals[50], 2)) + ")",
    #             'broad (sigma=' + str(round(sigma_vals[75], 2)) + ")"])
    # plt.show()
    cb1 = mpl.colorbar.ColorbarBase(axs[1], cmap=mpl.cm.viridis,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('sigma_G^2')
    plt.savefig("plots_20230206/zs_over_sigma.png", format='png')

    sigma2G_list = np.logspace(-8, -2, 30)
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]})
    colors = plt.cm.viridis(np.linspace(0, 1, 30))
    norm = mpl.colors.LogNorm(vmin=min(sigma2G_list), vmax=max(sigma2G_list))
    for i in range(30): #30
        axs[0].plot([power_all_zs(sg, 0.01, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=sigma2G_list[i],
                               sigma2R=0)[0]
                     for sg in sigma_vals],
                    [power_all_zs(sg, 0.01, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=sigma2G_list[i],
                               sigma2R=0)[1]
                     for sg in sigma_vals], color=colors[i])

    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("zs")
    axs[0].set_ylabel("power")
    # plt.legend(["point (sigma=0.01)", "narrow (sigma=" + str(round(sigma_vals[20], 2)) + ")",
    #             'broad (sigma=' + str(round(sigma_vals[50], 2)) + ")",
    #             'broad (sigma=' + str(round(sigma_vals[75], 2)) + ")"])
    # plt.show()
    cb1 = mpl.colorbar.ColorbarBase(axs[1], cmap=mpl.cm.viridis,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('sigma_G^2')
    plt.savefig("plots_20230206/power_over_zs.png", format='png')

    sigma2G_list = [1e-2]
    fig, axs = plt.subplots(1, 1)
    #colors = plt.cm.viridis(np.linspace(0, 1, 30))
    #norm = mpl.colors.LogNorm(vmin=min(sigma2G_list), vmax=max(sigma2G_list))
    for i in range(1):  # 30
        axs.plot([power_all_zs(sg, 0.01, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=sigma2G_list[i],
                                  sigma2R=0)[0]
                     for sg in sigma_vals],
                    [power_all_zs(sg, 0.01, sigma_vals, kappa_list, res_vals, pole_vals, sigma2G=sigma2G_list[i],
                                  sigma2R=0)[1]
                     for sg in sigma_vals], color=colors[i])

    axs.set_xscale("log")
    axs.set_yscale("log")
    axs.set_xlabel("zs")
    axs.set_ylabel("power")
    # plt.legend(["point (sigma=0.01)", "narrow (sigma=" + str(round(sigma_vals[20], 2)) + ")",
    #             'broad (sigma=' + str(round(sigma_vals[50], 2)) + ")",
    #             'broad (sigma=' + str(round(sigma_vals[75], 2)) + ")"])
    # plt.show()
    plt.savefig("plots_20230206/power_over_zs_sigma2G1e-2.png", format='png')



if __name__=="__main__":
    main()