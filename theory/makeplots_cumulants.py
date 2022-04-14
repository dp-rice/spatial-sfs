from matplotlib import pyplot as plt
import pandas as pd

def plot_cumulants(xvals,lists,labs,styles=['-',':','--'],xlab="spatial dispersion of sample (sigma)",ylab="value",outname="plt.png",xtrans=None,ytrans=None):
    plt.figure()
    for i in range(len(lists)):
        plt.plot(xvals,lists[i],styles[i],label=labs[i])
    plt.legend()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if xtrans is not None:
        plt.xscale(xtrans)
    if ytrans is not None:
        plt.yscale(ytrans)
    plt.savefig(outname)

def main():
    data = pd.read_csv('spatial_integrals.csv')
    sigma_list = data["sigma"]
    plot_cumulants(sigma_list, [data["u2_MC"], data["u2_GH"], data["u2_GQ"]],
                   ["u2 (Monte Carlo)", "u2 (Gauss-Hermite Quad)", "u2 (Gauss Quad)"],
                   outname="u2_allmethods.png")
    plot_cumulants(sigma_list, [data["u2_MC"], data["u2_GH"], data["u2_GQ"]],
                   ["u2 (Monte Carlo)", "u2 (Gauss-Hermite Quad)", "u2 (Gauss Quad)"],
                   outname="u2_allmethods_log.png", xtrans='log')
    plot_cumulants(sigma_list,[data["u2_MC"],data["u2_GH"],data["u2_GQ"]],
                   ["u2 (Monte Carlo)","u2 (Gauss-Hermite Quad)","u2 (Gauss Quad)"],
                   outname="u2_allmethods_loglog.png",xtrans='log',ytrans='log')

    plot_cumulants(sigma_list, [data["u3_MC"], data["u3_GQ"]],
                   ["u3 (Monte Carlo)", "u3 (Gauss Quad)"],
                   outname="u3_allmethods.png")
    plot_cumulants(sigma_list, [data["u3_MC"], data["u3_GQ"]],
                   ["u3 (Monte Carlo)", "u3 (Gauss Quad)"],
                   outname="u3_allmethods_log.png", xtrans='log')
    plot_cumulants(sigma_list, [data["u3_MC"], data["u3_GQ"]],
                   ["u3 (Monte Carlo)", "u3 (Gauss Quad)"],
                   outname="u3_allmethods_loglog.png", xtrans='log',ytrans='log')

    plot_cumulants(sigma_list, [data["u4_MC"], data["u4_GQ"]],
                   ["u4 (Monte Carlo)", "u4 (Gauss Quad)"],
                   outname="u4_allmethods.png")
    plot_cumulants(sigma_list, [data["u4_MC"], data["u4_GQ"]],
                   ["u4 (Monte Carlo)", "u4 (Gauss Quad)"],
                   outname="u4_allmethods_log.png", xtrans='log')
    plot_cumulants(sigma_list, [data["u4_MC"], data["u4_GQ"]],
                   ["u4 (Monte Carlo)", "u4 (Gauss Quad)"],
                   outname="u4_allmethods_loglog.png", xtrans='log', ytrans='log')

if __name__ == '__main__':
    main()