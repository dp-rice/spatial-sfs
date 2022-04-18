import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


data = pd.read_csv('pade_approx.csv')

sigma_list = [float(x) for x in data['sigma'].tolist()]
residues = [np.real(complex(x)) for x in data['residue'].tolist()]
poles = [np.real(complex(x)) for x in data['pole'].tolist()]

data["poly_type"] = data["m"].astype(str) + "_" + data["n"].astype(str)
categories = np.unique(data['poly_type'])
colors = np.linspace(0, 1, len(categories))
colordict = dict(zip(categories, colors))
data["Color"] = data['poly_type'].apply(lambda x: colordict[x])
labs = ['m=1, n=0','m=1, n=1','m=1, n=2','m=2, n=0','m=2, n=1','m=3, n=0']
plt.figure(1)
fig1 = plt.scatter(sigma_list,residues,s=0.3,c=data.Color)
plt.legend(handles=fig1.legend_elements()[0],labels=labs,
           title="polynomial degrees")
plt.xlabel("sigma")
plt.ylabel("residue of pole closest to zero")
plt.savefig("residues.png",dpi=300)

plt.figure(2)
fig2 = plt.scatter(sigma_list,poles,s=0.3,c=data.Color)
plt.legend(handles=fig2.legend_elements()[0],labels=labs,
           title="polynomial degrees")
plt.xlabel("sigma")
plt.ylabel("pole closest to zero")
plt.savefig("poles.png",dpi=300)

plt.figure(3)
fig1 = plt.scatter(sigma_list,residues,s=0.3,c=data.Color)
plt.xscale('log')
plt.yscale('symlog')
plt.legend(handles=fig1.legend_elements()[0],labels=labs,
           title="polynomial degrees")
plt.xlabel("sigma")
plt.ylabel("residue of pole closest to zero")
plt.savefig("residues_log.png",dpi=300)

plt.figure(4)
fig2 = plt.scatter(sigma_list,poles,s=0.3,c=data.Color)
plt.legend(handles=fig2.legend_elements()[0],labels=labs,
           title="polynomial degrees")
plt.xlabel("sigma")
plt.ylabel("pole closest to zero")
plt.xscale('log')
plt.yscale('symlog')
plt.savefig("poles_log.png",dpi=300)

