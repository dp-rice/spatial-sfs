import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read in cumulants
data = pd.read_csv("numerics_snakemake/spatial_integrals_dim2.csv")
data


def calc_H_cumulants(kappa_list, s, mu=1e-8, D=1, d=1,N=10000):
    H_list = [(2*mu/s)*(1-(k/(N*s*np.sqrt(D/s)**d))) for k in kappa_list]
    return (H_list)

font = {'size': 18}

plt.rc('font', **font)

sigma_list = data['sigma'].tolist()
colors = sns.color_palette("dark", 3)
# N_vals = [0.5,1,10,100,1000]

# fig, axs = plt.subplots(1,5,figsize=(20,4))
# for i in range(5):
#     axs[i].plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.001, d=2, N=N_vals[i]), color=colors[0])
#     axs[i].plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.01, d=2, N=N_vals[i]), color=colors[1])
#     axs[i].plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.1, d=2, N=N_vals[i]), color=colors[2])
#     axs[i].legend(labels=['0.001', '0.01', '0.1'], title="s")
#     # print(calc_H_cumulants(data['u2_GQ'].tolist(), 0.001, d=2, N=N_vals[i]))
#     axs[i].set_xscale("log")
#     axs[i].set_yscale("log")
#     axs[i].set_title("N="+str(N_vals[i]))
#     axs[i].set_xlabel("sigma")
#     axs[i].set_ylabel("value")
#     axs[i].set_ylim(1e-9,1e-4)

# fig.set_figheight(3.622)
# fig.set_figwidth(5.748)

N_vals = [0.5,100]
fig,ax=plt.subplots(1,1)
ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.001, d=2, N=N_vals[0]), color=colors[0])
ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.01, d=2, N=N_vals[0]), color=colors[1])
ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.1, d=2, N=N_vals[0]), color=colors[2])
ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.001, d=2, N=N_vals[1]), color=colors[0],linestyle='dashed')
ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.01, d=2, N=N_vals[1]), color=colors[1],linestyle='dashed')
ax.plot(sigma_list, calc_H_cumulants(data['u2_GQ'].tolist(), 0.1, d=2, N=N_vals[1]), color=colors[2],linestyle='dashed')
# ax.legend(labels=['0.001', '0.01', '0.1'], title="s")
ax.set_title("Expected heterozygosity")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("sigma")
ax.set_ylabel("value")
ax.set_ylim(1e-9,1e-4)

dummy_lines = []
dummy_lines.append(ax.plot([],[], c="black", ls = 'solid')[0])
dummy_lines.append(ax.plot([],[], c="black", ls = 'dashed')[0])

legend1=ax.legend(labels=['0.001', '0.01', '0.1'], title="s",loc=3)
legend2=ax.legend([dummy_lines[i] for i in [0,1]], ["low","high"],title="Population density", loc=4)
ax.add_artist(legend1)


fig.set_figheight(3.622)
fig.set_figwidth(5.748)

plt.tight_layout()
# plt.show()
#plt.savefig("exp_het_overN_2d_20221208.pdf")
plt.savefig("exp_het_20230228.pdf")
