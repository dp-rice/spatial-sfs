from scipy.special import erfinv
from scipy.special import loggamma
from scipy.stats import gamma
from scipy.stats import nbinom
from scipy import special
from scipy import stats
import numpy as np
import sys
import math
from matplotlib import pyplot as plt
import seaborn as sns
import random
from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter
from scipy.special import binom
from scipy.fft import fft2
random.seed(1234)

def get_all_indices(rows,cols):
    indices = []
    for i in range(rows):
        for j in range(cols):
            indices.append((i, j))
    return indices

def simulate_WF(m,dims,pop_size,s,num_intervals):
    # create array of (num_intervals) matrices each with dimensions given by dims
    output = np.zeros(tuple([num_intervals]) + dims)
    # for positive value of s, interval size is 1/s
    if s > 0:
        interval = int(1 / s)
    else: # if s=0, interval=100
        interval = 100
    # number of generations to simulate
    num_gens = interval * num_intervals
    # create array to store frequencies
    f = np.zeros(dims)
    # choose random deme & initialize variant there
    # *this is changed from old version - was always initialized in the same place*
    rand_x, rand_y = random.randint(0, dims[0] - 1), random.randint(0, dims[1] - 1)
    # initialize at frequency 1/N
    f[rand_x, rand_y] = 1 / pop_size
    ## pre-allocate list of when reset happens
    reset_gens = []
    for i in range(num_gens):
        # Wright-Fisher diffusion w/Stepping Stone migration
        df = - s * f * (1 - f) + m * laplace(f, mode='wrap')
        # bounds allele frequencies in [0,1]
        p = np.clip(a=f + df, a_min=0, a_max=1)
        # binomial sampling from frequencies
        new_f = np.random.binomial(pop_size, p) / pop_size
        # Check for extinct allele
        if np.all(new_f == 0):
            ## if extinct, store previous freqs and reset
            reset_gens.append(i)
            ## Initialize new allele
            rand_x, rand_y = random.randint(0, dims[0] - 1), random.randint(0, dims[1] - 1)
            new_f[rand_x, rand_y] = 1 / pop_size
        # Assign next frequencies
        f = new_f
        # store freqs every 1/s
        if (i + 1) % interval == 0:
            output[i // interval] = f
    return output, reset_gens


# noinspection PyTypeChecker
def get_exp_values(s_list=[1e-3,1e-2,1e-1],L=1,l=1,m=1e-3,Nd=1000,num_intervals=5000):
    #################################
    # want to output:
    ## list of E[|C(k)|] **list1**
    ## list of |E[C(k)]|^2 **list2**
    ## list of E[|C(k)|^2] **list3**
    ## list of times to extinction for each s
    # for range of s values & all relevant k values (except times)
    #################################

    dims = (int(L / l), int(L / l))  # dimensions for simulation
    time_ext_list = []

    k_list = get_all_indices(L, L)

    list1 = np.zeros((len(s_list), len(k_list)))
    list2 = np.zeros((len(s_list), len(k_list)))
    list3 = np.zeros((len(s_list), len(k_list)))
    time_ext_list = []
    for i in range(len(s_list)):
        # run simulation
        # print("*** starting simulation ***")
        sim_output, sim_gens = simulate_WF(m=m, dims=dims, pop_size=Nd, s=s_list[i], num_intervals=num_intervals)
        # print("*** finished simulation ***")
        prev_gen=0
        time_ext = []
        for g in sim_gens:
            diff = g - prev_gen
            time_ext.append(diff)
            prev_gen = g
        time_ext_list.append(time_ext)
        dft = np.zeros(tuple([num_intervals]) + dims)
        dft_abs = np.zeros(tuple([num_intervals]) + dims)
        for t in range(num_intervals):
            dft[t] = fft2(sim_output[t])
            dft_abs[t] = np.abs(fft2(sim_output[t]))
        dft_mean_abs = np.mean(dft_abs, axis=0)  # take mean over timesteps
        dft_mean_square_abs = np.mean(dft_abs ** 2, axis=0)
        dft_mean = np.mean(dft,axis=0)
        for j in range(len(k_list)):
            list1[i,j] = dft_mean_abs[k_list[j][0],k_list[j][1]]
            list2[i,j] = np.abs(dft_mean[k_list[j][0],k_list[j][1]])**2
            list3[i,j] = dft_mean_square_abs[k_list[j][0],k_list[j][1]]

    time_ext_arr = np.array(time_ext_list, dtype=object)
    time_means = [np.mean(data) for data in time_ext_arr]

    # list1 = np.insert(list1, 0, k_list, axis=0)
    # list1 = np.insert(list1, 0, s_list.insert(0,np.nan), axis=1)
    #
    # list2 = np.insert(list2, 0, k_list, axis=0)
    # list2 = np.insert(list2, 0, s_list.insert(0,np.nan), axis=1)
    #
    # list3 = np.insert(list3, 0, k_list, axis=0)
    # list3 = np.insert(list3, 0, s_list.insert(0,np.nan), axis=1)
    #
    # time_means = np.insert(time_means,0,s_list,axis=0)



    return list1,list2,list3,time_means,k_list,s_list

def main():
    L_list = [1,2,10]
    for Lval in L_list:
        L = Lval # later iterate across L's
        l1,l2,l3,times,k_list,s_list = get_exp_values(L=L)
        np.savetxt('L'+str(L)+'_list1.csv', l1, delimiter=',')
        np.savetxt('L'+str(L)+'_list2.csv', l2, delimiter=',')
        np.savetxt('L'+str(L)+'_list3.csv', l3, delimiter=',')
        np.savetxt('L'+str(L)+'_times.csv', times, delimiter=',')
        np.savetxt('L' + str(L) + '_k_list.csv', k_list, delimiter=',')
        np.savetxt('L' + str(L) + '_s_list.csv', s_list, delimiter=',')

    # L = 3  # later iterate across L's
    # l1, l2, l3, times, k_list, s_list = get_exp_values(L=L)
    # np.savetxt('L' + str(L) + '_list1.csv', l1, delimiter=',')
    # np.savetxt('L' + str(L) + '_list2.csv', l2, delimiter=',')
    # np.savetxt('L' + str(L) + '_list3.csv', l3, delimiter=',')
    # np.savetxt('L' + str(L) + '_times.csv', times, delimiter=',')
    # np.savetxt('L' + str(L) + '_k_list.csv', k_list, delimiter=',')
    # np.savetxt('L' + str(L) + '_s_list.csv', s_list, delimiter=',')

if __name__ == '__main__':
    main()
