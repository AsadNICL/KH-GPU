# params.jl
# Define all simulation parameters as arrays.
# - If a parameter should vary between runs, list multiple values.
# - If a parameter is fixed, give a single value (in a 1-element array).
# When running, parameters with multiple values are paired by index,
# while single-valued parameters are reused for all runs.
# So, if a parameter is given multiple values, all other parameters must either have the same length or length 1

nu_values       = [2e-5, 4e-5, 6e-5, 8e-5, 1e-4]     # Viscosity
gred_values      = [1.0]        # Fixed reduced gravity for delta comparison
delta_values      = [1.0]  # Shear layer thickness values to compare
Pr_values       = [700]     # Prandtl number
Lz_values       = [10]       # Domain height
Lx_values       = [400]      # Domain length in the x-direction
Nx_values       = [4000]  # Number of grid points in x-direction
Nz_values       = [200]    # Number of grid points in z-direction
efgm_values     = [1e-4]    # Fraction of background energy in the initial perturbation
enoise_values   = [1e-6]    # Noise amplitude in initial conditions         
T_values        = [35]      # Simulation duration
initial_tilt_angle_values = [10]  # Initial tilt angle of the domain relative to horizontal (degrees)
rotate_back_to_horizontal_values = [true]  # Whether to rotate back to horizontal after the reset time
angle_indep_rot_param_values = [14sind(10)]  # Effective rotation trigger parameter T, applied at t = T / sin(theta0)
