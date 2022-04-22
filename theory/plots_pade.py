import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def plot_scatter(df,y,x="sigma",outname="fig.png",xtrans=None,ytrans=None):
    sns.lmplot(x, y, data=df, hue='poly_type', fit_reg=False,scatter_kws={"s": 2},legend=False)
    if xtrans is not None:
        plt.xscale(xtrans)
    if ytrans is not None:
        plt.yscale(ytrans)
    plt.legend(markerscale=5)
    plt.savefig(outname,dpi=300)

def main():
    data = pd.read_csv('pade_approx.csv')
    sigma_list = [float(x) for x in data['sigma'].tolist()]
    residues = [np.real(complex(x)) for x in data['residue'].tolist()]
    poles = [np.real(complex(x)) for x in data['pole'].tolist()]
    residues_im = [np.imag(complex(x)) for x in data['residue'].tolist()]
    poles_im = [np.imag(complex(x)) for x in data['pole'].tolist()]

    data["poly_type"] = data["m"].astype(str) + "_" + data["n"].astype(str)
    poly_type = data["poly_type"].tolist()

    # df = pd.DataFrame(dict(sigma=sigma_list, residues=residues, poles=poles,poly_type=poly_type))
    # # df.to_csv("res_pole_values.csv")
    # df['poles'].replace('',np.nan,inplace=True)
    # df.dropna(subset='poles',inplace=True)
    # df['residues'].replace('', np.nan, inplace=True)
    # df.dropna(subset='residues', inplace=True)

    df = pd.DataFrame(dict(sigma=sigma_list, residues=residues, poly_type=poly_type))
    plot_scatter(df,y='residues',outname="residues.png")
    plot_scatter(df, y='residues', outname="residues_log.png",xtrans='log',ytrans='symlog')

    df = pd.DataFrame(dict(sigma=sigma_list, poles=poles, poly_type=poly_type))
    plot_scatter(df,y='poles',outname="poles.png")
    plot_scatter(df, y='poles', outname="poles_log.png",xtrans='log',ytrans='symlog')

    df = pd.DataFrame(dict(sigma=sigma_list, residues_im=residues_im, poly_type=poly_type))
    plot_scatter(df,y='residues_im',outname="residues_im.png")
    plot_scatter(df, y='residues_im', outname="residues_log_im.png",xtrans='log',ytrans='symlog')

    df = pd.DataFrame(dict(sigma=sigma_list, poles_im=poles_im, poly_type=poly_type))
    plot_scatter(df,y='poles_im',outname="poles_im.png")
    plot_scatter(df, y='poles_im', outname="poles_log_im.png",xtrans='log',ytrans='symlog')


if __name__ == '__main__':
    main()


