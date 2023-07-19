import numpy as np
import argparse
import random
from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter
from scipy.special import binom
from scipy.fft import fft2

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
            # print("Interval: " +str(i // interval))
    return output, reset_gens


def get_exp_values(sim_output,sim_gens,s,L=1,l=1,m=1e-3,Nd=1000,num_intervals=5000):
    dims = (int(L / l), int(L / l))  # dimensions for simulation
    k_list = get_all_indices(L, L)
    mean_list = np.zeros(len(k_list))
    mean_square_list = np.zeros(len(k_list))
    prev_gen=0
    time_ext = []
    for g in sim_gens:
        diff = g - prev_gen
        time_ext.append(diff)
        prev_gen = g
    dft_abs = np.zeros(tuple([num_intervals]) + dims)
    for t in range(num_intervals):
        dft_abs[t] = np.abs(fft2(sim_output[t]))
    dft_mean_abs = np.mean(dft_abs, axis=0)  # take mean over timesteps
    dft_mean_square_abs = np.mean(dft_abs ** 2, axis=0)
    for j in range(len(k_list)):
        mean_list[j] = dft_mean_abs[k_list[j][0],k_list[j][1]]
        mean_square_list[j] = dft_mean_square_abs[k_list[j][0],k_list[j][1]]
    time_ext_arr = np.array(time_ext, dtype=object)
    time_mean = np.mean(time_ext_arr)
    k_list_flat = [str(kvals[0])+"_"+str(kvals[1]) for kvals in k_list]
    res_all = np.vstack((k_list_flat,mean_list,mean_square_list,np.repeat(time_mean,len(k_list))))
    return res_all

def sample_f(f):
    # values of sigma to sample - from 1 to width of grid
    sig_arr =  np.arange(f.shape[-1])+1
    # create array with copy of f for each entry in sig_arr
    f_filt = np.zeros((tuple([len(sig_arr)])+f.shape))
    # iterate over values of sigma
    for i,sig in enumerate(sig_arr):
        # for each entry in matrix
        for j in range(f.shape[0]):
            for k in range(f.shape[1]):
                # frequency stored in f_filt is gaussian filter applied to frequency in f
                # sigma defines width of kernel
                f_filt[i,j,k]=gaussian_filter(f[j,k],sigma=sig,mode="wrap")
    return f_filt

def freq_sfs(f,n):
    # create array to store entries of sfs
    sfs = np.zeros(tuple([n + 1]) + f.shape)
    # iterate over entries in sfs
    for j in range(n + 1):
        # calculate SFS entry from frequency using binomial sampling
        sfs[j] = binom(n, j) * f ** j * (1 - f) ** (n - j)
    return sfs

def calc_sfs(f,n):
    # store list of sigma values
    sig_list = np.arange(f.shape[-1]) + 1
    # sample from f with range of sigma
    f_filt = sample_f(f)
    # j is allele count, range 1 to n
    j = np.arange(1,n+1) # ignore j=0
    # average sampled f over intervals,reps, dims
    sfs = freq_sfs(f_filt,n).T
    # avg over both spatial dimensions & generations
    sfs_avg = np.mean(sfs,axis=(0,1,2))
    # print(j.T)
    # print(sfs_avg)
    sfs_avg = np.vstack(sfs_avg)[:,1:]
    res_all = np.column_stack((sig_list,sfs_avg))
    return res_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=float, help="symmetric migration rate", default=1e-1)
    parser.add_argument("-L", type=int, help="width of spatial lattice", default=2)
    parser.add_argument("--pop_size", type=int, help="number of individuals in each deme", default=1e3)
    parser.add_argument("-s", type=float, help="selection coefficient", default=1e-2)
    parser.add_argument("-l", type=float, help="selection coefficient", default=1.0)
    parser.add_argument("-n", type=int, help="sample size", default=100)
    parser.add_argument("--num_intervals", type=int, help="number of intervals to store", default=100)
    parser.add_argument("--outdir", type=str, help="output path", default="output")
    parser.add_argument("--seed", type=int, help="random seed", default=1234)
    parser.add_argument("--check_sims",action='store_true')
    parser.add_argument("--calc_sfs", action='store_true')
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)

    # run simulation
    dims = (args.L,args.L)
    output, reset_gens = simulate_WF(args.m, dims, args.pop_size, args.s, args.num_intervals)

    # get values for sanity checks
    if args.check_sims is True:
        test_vals = get_exp_values(output,reset_gens,0.1,args.L,args.l,args.m,args.pop_size,args.num_intervals)
        test_vals_filename = args.outdir+'/test_vals_L'+str(args.L)+"_l"+str(args.l)+"_m"+str(args.m)+"_s"+str(args.s)+"_N"+str(args.pop_size)+"_numint"+str(args.num_intervals)+".csv"
        np.savetxt(test_vals_filename,test_vals,delimiter=',',fmt = '%s')

    # calc SFS with sampling
    if args.calc_sfs is True:
        sfs_vals = calc_sfs(output,args.n)
        sfs_vals_filename = args.outdir + '/sfs_vals_L' + str(args.L) + "_l" + str(args.l) + "_n"+str(args.n)+"_m" + str(
            args.m) + "_s"+str(args.s)+"_N" + str(args.pop_size) + "_numint" + str(args.num_intervals) + ".csv"
        np.savetxt(sfs_vals_filename,sfs_vals,delimiter=',',fmt='%s')

if __name__ == '__main__':
    main()
