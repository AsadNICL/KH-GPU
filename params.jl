# params.jl
# Define all simulation parameters as arrays.
# - If a parameter should vary between runs, list multiple values.
# - If a parameter is fixed, give a single value (in a 1-element array).
# When running, parameters with multiple values are paired by index,
# while single-valued parameters are reused for all runs.
# So, if a parameter is given multiple values, all other parameters must either have the same length or length 1

nu_values       = [1/8e5]     # Viscosity
gred_values      = [1.0,2.0]        # Fixed reduced gravity for delta comparison
delta_values      = [1.0]  # Shear layer thickness values to compare
Pr_values       = [700]     # Prandtl number
Lz_values       = [10]       # Domain height
Nx_values       = [4000]  # Number of grid points in x-direction
Nz_values       = [200]    # Number of grid points in z-direction
efgm_values     = [1e-4]    # Fraction of background energy in the initial perturbation
enoise_values   = [1e-6]    # Noise amplitude in initial conditions         
T_values        = [60]      # Simulation duration
