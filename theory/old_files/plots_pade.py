import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from pade_approx import get_pade_poly
import ast

def plot_scatter(df,y,x="sigma",outname="fig.png",xtrans=None,ytrans=None):
    sns.lmplot(x, y, data=df, hue='poly_type', fit_reg=False,scatter_kws={"s": 2},legend=False)
    if xtrans is not None:
        plt.xscale(xtrans)
    if ytrans is not None:
        plt.yscale(ytrans)
    plt.legend(markerscale=5)
    plt.savefig(outname,dpi=300)

def main():
    data = pd.read_csv('pade_approx_2d.csv')
    sigma_list = [float(x) for x in data['sigma'].tolist()]
    residues = [np.real(complex(x)) for x in data['residue'].tolist()]
    poles = [np.real(complex(x)) for x in data['pole'].tolist()]
    # residues_im = [np.imag(complex(x)) for x in data['residue'].tolist()]
    # poles_im = [np.imag(complex(x)) for x in data['pole'].tolist()]
    rem=[np.real(complex(x)) for x in data['remainder'].tolist()]
    data["poly_type"] = data["m"].astype(str) + "_" + data["n"].astype(str)
    poly_type = data["poly_type"].tolist()
    #error_next = data["error_next"].tolist()
    #rel_error = data["rel_err"].tolist()
    pole_2 = data["pole_2"].tolist()
    res_2 = data["res_2"].tolist()
    df = pd.DataFrame(dict(sigma=sigma_list, residues=residues, poles=poles,poly_type=poly_type,pole_2=pole_2,res_2=res_2,remainder=rem))#,error_next=error_next,rel_error=rel_error
    df.to_csv("cleaned_data_2d.csv")

    plot=False

    if plot==True:
        sigma_cases=[0.01,1.0109009009009,10.019009009009,100]
        # print(len(data.sigma))
        # k=0
        colors=["red","blue","green","orange","purple","yellow"]
        filenames=["pade_rationals"+str(s)+".png" for s in sigma_cases]
        # print(filenames)
        j = 0
        for s in sigma_cases:
            plt.figure()
            tab_s = data[abs(data.sigma-s)<10E-14]
            k = 0
            for row in tab_s.itertuples():

                if row[tab_s.columns.get_loc('m') + 1] > 0:
                    p, q = get_pade_poly(tab_s, row[tab_s.columns.get_loc('m') + 1], row[tab_s.columns.get_loc('n') + 1])
                    x_lower=0
                    x_upper=round(np.real(complex(row[tab_s.columns.get_loc('pole')+1])))+1
                    x_range=np.linspace(x_lower,x_upper,100)
                    plt.plot(x_range, [p(x)/q(x) for x in x_range], label=str(row[tab_s.columns.get_loc('poly_type') + 1]),color=colors[k])
                    plt.xlabel("x")
                    plt.ylabel("R(x)")
                    plt.legend(loc="lower left")
                    plt.axvline(x=np.real(complex(row[tab_s.columns.get_loc('pole')+1])),linestyle='dotted',color=colors[k])
                    plt.ylim(-500,500)
                    k+=1
            plt.title("sigma = "+str(round(s,4)))
            plt.savefig(fname=filenames[j])
            j+=1


    # df['poles'].replace('',np.nan,inplace=True)
    # df.dropna(subset='poles',inplace=True)
    # df['residues'].replace('', np.nan, inplace=True)
    # df.dropna(subset='residues', inplace=True)

    # df = pd.DataFrame(dict(sigma=sigma_list, residues=residues, poly_type=poly_type))
    # plot_scatter(df,y='residues',outname="residues.png")
    # plot_scatter(df, y='residues', outname="residues_log.png",xtrans='log',ytrans='symlog')
    #
    # df = pd.DataFrame(dict(sigma=sigma_list, poles=poles, poly_type=poly_type))
    # plot_scatter(df,y='poles',outname="poles.png")
    # plot_scatter(df, y='poles', outname="poles_log.png",xtrans='log',ytrans='symlog')

    # df = pd.DataFrame(dict(sigma=sigma_list, residues_im=residues_im, poly_type=poly_type))
    # plot_scatter(df,y='residues_im',outname="residues_im.png")
    # plot_scatter(df, y='residues_im', outname="residues_log_im_test4.png",xtrans='log',ytrans='symlog')

    # df = pd.DataFrame(dict(sigma=sigma_list, poles_im=poles_im, poly_type=poly_type))
    # plot_scatter(df,y='poles_im',outname="poles_im.png")
    # plot_scatter(df, y='poles_im', outname="poles_log_im_test4.png",xtrans='log',ytrans='symlog')


if __name__ == '__main__':
    main()


