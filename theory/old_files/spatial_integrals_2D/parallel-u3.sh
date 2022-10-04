#!/usr/bin/env bash
#SBATCH -J parallel-u3
#SBATCH -N 5
#SBATCH -n 10
#SBATCH --mem=3G

module load python
conda activate parallel
mpiexec python spatial_integrals_2D.py
