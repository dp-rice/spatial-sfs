import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read in cumulants
data = pd.read_csv("numerics_snakemake/spatial_integrals_dim2.csv")
data


def calc_H_cumulants(kappa_list, s, mu=1e-8, D=1, d=1,N=10000):
    #scale_term = (2 * mu) / (s * s * (np.sqrt(D / s)**d))
    H_list = [(2*mu/s)*(1-(1/(N*s*np.sqrt(D/s)**d)*k)) for k in kappa_list]
    return (H_list)

font = {'size': 18}

plt.rc('font', **font)

sigma_list = data['sigma'].tolist()
# H_list = calc_H_cumulants(data['u2_GQ'].tolist(),0.1)

colors = sns.color_palette("dark", 3)

# code below was used to generate poster figures
fig, ax = plt.subplots()
ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.001, d=2), color=colors[0])#,linewidth=24)
ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.01, d=2), color=colors[1])#,linewidth=24)
ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.1, d=2), color=colors[2])#,linewidth=24)
# ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.01, d=2), color=colors[3])
ax.legend(labels=['0.001', '0.01', '0.1'], title="s")
plt.xscale("log")
plt.yscale("log")
ax.set_title("Expected heterozygosity")

ax.set_xlabel("sigma")
ax.set_ylabel("value")

# fig.set_figheight(3.622)
# fig.set_figwidth(5.748)

plt.savefig("exp_het_2d_20221208.pdf")


data = pd.read_csv("numerics_snakemake/spatial_integrals_dim1.csv")
fig, ax = plt.subplots()
ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.001, d=1), color=colors[0])#,linewidth=24)
ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.01, d=1), color=colors[1])#,linewidth=24)
ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.1, d=1), color=colors[2])#,linewidth=24)
# ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.01, d=2), color=colors[3])
ax.legend(labels=['0.001', '0.01', '0.1'], title="s")
plt.xscale("log")
plt.yscale("log")
ax.set_title("Expected heterozygosity")

ax.set_xlabel("sigma")
ax.set_ylabel("value")

# fig.set_figheight(3.622)
# fig.set_figwidth(5.748)

plt.savefig("exp_het_1d_20221208.pdf")
