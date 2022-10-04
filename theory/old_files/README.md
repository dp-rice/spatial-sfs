## Code & documentation for numerical steps in approximating the SFS

Maggie Steiner

Important files:
* `spatial_integrals.py`: functions to compute cumulants using Monte Carlo methods and Gaussian quadrature, outputs `spatial_integrals.csv`
* `makeplots_cumulants.py`: script to generate plots in `theory/plots_cumulants`
* `pade_approx.py`: functions to compute Pade tables and calculate poles/residues
* `pade_approx.csv` and `spatial_integrals.csv`: results files from respective scripts
* `plots_pade.py`: creates `*.png` files with poles and residues (real + imaginary parts) and saves a results file as `res_pole_values.csv`
* `numerical_solution_fipy.py`: script to identify singularities by solving using the Finite Volume Method as implemented in `FiPy`
