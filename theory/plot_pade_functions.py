import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def plot_scatter(df, xmin=0,xmax=20, poly_type_list=['1_1'], outname="fig.png",xtrans=None,ytrans=None):#sigma,
    plt.figure()
    xvals = np.linspace(xmin,xmax)
    for i in range(len(poly_type_list)):
        p = df.loc[(df["pt"]==poly_type_list[i]),"p"].tolist()[0]#p = df.loc[(df["sigma"]==sigma) & (df["pt"]==poly_type_list[i]),"p"].tolist()[0]
        p = p.replace("[","")
        p = p.replace("]","").split()
        p = [float(x) for x in p]
        p = np.poly1d(p)
        q = df.loc[ (df["pt"] == poly_type_list[i]), "q"].tolist()[0]#df.loc[(df["sigma"] == sigma) & (df["pt"] == poly_type_list[i]), "q"].tolist()[0]
        q = q.replace("[", "")
        q = q.replace("]", "").split()
        q = [float(x) for x in q]
        q = np.poly1d(q)
        # print(p)
        # print(q)
        y = [np.polyval(p,x)/np.polyval(q,x) for x in xvals]
        plt.plot(xvals,y,label=poly_type_list[i])
        # sns.lmplot(xvals, y, data=df, hue='poly_type', fit_reg=False,scatter_kws={"s": 2},legend=False)
    if xtrans is not None:
        plt.xscale(xtrans)
    if ytrans is not None:
        plt.yscale(ytrans)
    plt.legend()
    # plt.title("sigma: "+str(sigma))
    plt.savefig(outname,dpi=300)

def main():
    data = pd.read_csv('pade_approx_test.csv')#('pade_approx.csv')
    # sigma_list = [float(x) for x in data['sigma'].tolist()]
    p_coefs = [p for p in data['p'].tolist()]
    q_coefs = [q for q in data['q'].tolist()]
    data["poly_type"] = data["m"].astype(str) + "_" + data["n"].astype(str)
    poly_type = data["poly_type"].tolist()
    df = pd.DataFrame(dict(p=p_coefs, q=q_coefs,pt=poly_type))#sigma=sigma_list,
    pl =df['pt'].unique().tolist()
    plot_scatter(df,0,2,pl,outname="test_pade.png")


    # plot_scatter(df, 0.01,0,8,poly_type_list=['1_0','1_1','1_2'],outname="sigma01.png")
    #
    # idx = df['sigma'].sub(1).abs().idxmin()
    # sig1 = df.loc[idx, 'sigma']
    # plot_scatter(df, sig1, 0,8,poly_type_list=['1_1', '1_2'], outname="sigma1_v2.png")
    #
    # idx = df['sigma'].sub(10).abs().idxmin()
    # sig10 = df.loc[idx, 'sigma']
    # plot_scatter(df, sig10, 0,14,poly_type_list=['1_0', '1_1', '1_2'], outname="sigma10.png")

if __name__ == '__main__':
    main()


