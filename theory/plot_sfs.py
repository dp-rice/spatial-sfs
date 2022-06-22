from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math

def sfs(y,t,a):

    return (y**(t-1))*math.exp(-1*a*y)

def sfs_plot(y_list,df,sigma_vals):
    fig,ax = plt.subplots()
    for i in range(len(sigma_vals)):
        idx = df['sigma'].sub(sigma_vals[i]).abs().idxmin()
        res = np.real(complex(df.loc[idx,'residue']))
        pol = np.real(complex(df.loc[idx,'pole']))
        p_list = [sfs(y,res,pol) for y in y_list]
        ax.plot(y_list,p_list,label=str(sigma_vals[i]))
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    # plt.show()



def main():
    df = pd.read_csv('pade_approx.csv')
    df_sub = df.loc[(df['m']==1) & (df['n']==0),['sigma','pole','residue']]
    y_list = np.linspace(0,100,100)

    sigma_to_plot = [1,10,100]



    # print(df_sub.loc[(df_sub['sigma']==)])
    sfs_plot(y_list,df_sub,sigma_to_plot)




if __name__ == '__main__':
    main()