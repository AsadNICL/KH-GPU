using Base.Threads
using Dates
using CUDA
using JLD2
ENV["GKSwstype"] = "100"
using Plots
gr()
using Measures
using Printf
using Oceananigans
using Oceananigans.OutputWriters
using Interpolations
using Statistics
using LinearAlgebra
using SpecialFunctions


@info "-----------------------------------------------------------------------------------------------"
@info "-----------------------------------------------------------------------------------------------"

# Log GPU information
CUDA.NVML.nvmlInit()
handle = CUDA.NVML.device()
props = CUDA.device()
@info "GPU name: $(CUDA.name(props))"
sm_count = CUDA.attribute(props, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
@info "Streaming Multiprocessors: $sm_count"

# Assumes Julia was launched from the repo root (the directory containing this script)
include("./code/SSF.jl")
include("./code/ddz.jl")
include("./code/ddz2.jl")
include("./code/ddz4.jl")
include("./lin_stab_from_params.jl")
include("./KH_from_params.jl")
include("./plot_KH.jl")
include("./plot_KH_b_only.jl")  # alternative lighter plotting (b only)
include("./plot_Ri.jl")
include("./plot_Ri_vs_Re.jl")
include("./plot_velocity_magnitude.jl")
include("./plot_velocity_tanh_fit.jl")
include("./plot_dudz.jl")

# Toggle whether to visualize the pointwise Ri heatmap animation during each run.
const generate_Ri_animation = false
# Toggle whether to visualize the velocity magnitude heatmap animation during each run.
const generate_velocity_magnitude_animation = false
# Toggle whether to fit and plot the velocity tanh profile during each run.
const generate_velocity_tanh_fit = false
# Toggle whether to visualize du/dz heatmap animation during each run.
const generate_dudz_animation = false
# Toggle whether to generate the aggregate Ri vs Re plot after runs.
const generate_Ri_Re_plot = true
# Toggle whether to save the animation during each run.
const save_KH_animation = true

# Read parameter arrays
include("./params.jl")
params = Dict(
    :nu => nu_values,
    :gred => gred_values,
    :delta => delta_values,
    :Pr => Pr_values,
    :Lz => Lz_values,
    :Nx => Nx_values,
    :Nz => Nz_values,
    :efgm => efgm_values,
    :enoise => enoise_values,
    :T => T_values,
)

# Choose the parameter to compare in the aggregate Ri vs Re plot.
function infer_sweep_param(params)
    varying = filter(k -> length(params[k]) > 1, (:gred, :delta))
    if length(varying) == 1
        return varying[1]
    elseif length(varying) == 2
        @warn "Both gred and delta vary; labeling with both parameters."
        return :both
    else
        return :delta
    end
end

function make_run_label(gred, delta)
    if sweep_param == :gred
        return @sprintf("gred=%.3f", gred)
    elseif sweep_param == :delta
        return @sprintf("delta=%.3f", delta)
    else
        return @sprintf("gred=%.3f, delta=%.3f", gred, delta)
    end
end

function make_ri_re_filename(gred, delta)
    if sweep_param == :gred
        return @sprintf("Ri_vs_inv_sq_Re_gred=%.3f.png", gred)
    elseif sweep_param == :delta
        return @sprintf("Ri_vs_inv_sq_Re_delta=%.3f.png", delta)
    else
        return @sprintf("Ri_vs_Re_gred=%.3f_delta=%.3f.png", gred, delta)
    end
end

const sweep_param = infer_sweep_param(params)
@info "Sweep parameter set to $(sweep_param)"

# Find the maximum length across all parameter arrays
nruns = maximum(length.(values(params)))

# Validate: each array must have length == 1 or == nruns
for (name, arr) in params
    if !(length(arr) == 1 || length(arr) == nruns)
        error("Parameter $name has length $(length(arr)), which is not 1 or $nruns")
    end
end

# Helper to get i-th value or repeat a single value
getval(a, i) = length(a) == 1 ? a[1] : a[i]

function jld2_has_key(filename::String, key::String)
    if !isfile(filename)
        return false
    end
    try
        jldopen(filename, "r") do f
            return haskey(f, key)
        end
    catch
        return false
    end
end

# Main loop
runs_start_time = time_ns()
all_Reynolds = Vector{Vector{Float64}}()
all_Rig = Vector{Vector{Float64}}()
all_labels = String[]
all_theory_coeffs = Float64[]
for irun in 1:nruns
    # Extract parameters for this run 
    @info "STARTING RUN $irun of $nruns ..."
    nu = getval(params[:nu], irun)
    gred = getval(params[:gred], irun)
    delta = getval(params[:delta], irun)
    Pr = getval(params[:Pr], irun)
    Lz = getval(params[:Lz], irun)
    Nx = getval(params[:Nx], irun)
    Nz = getval(params[:Nz], irun)
    efgm = getval(params[:efgm], irun)
    enoise = getval(params[:enoise], irun)
    T = getval(params[:T], irun)

    # Set output paths
    today_str = Dates.format(today(), "yyyy-mm-dd")
    today_str = replace(today_str, "-" => "") # to match the required format
    base_path = pwd() # should be the KH_GPU_sweep directory

    out_path = joinpath(base_path, "out")
    lin_stab_out_name = @sprintf("KH_lin_stab_nu=%.1e_gred=%0.2f_delta=%0.2f_Pr=%d_Lz=%d_efgm=%.1e",
        nu, gred, delta, Pr, Lz, efgm)
    lin_stab_out_file = joinpath(out_path, lin_stab_out_name)
    dns_out_name = @sprintf("KH_%s_nu=%.1e_gred=%0.2f_delta=%0.2f_Pr=%d_Lz=%d_Nx=%d_Nz=%d_efgm=%.1e_enoise=%.1e_T=%d",
        today_str, nu, gred, delta, Pr, Lz, Nx, Nz, efgm, enoise, T)
    dns_out_file = joinpath(out_path, dns_out_name)

    lin_stab_filename = lin_stab_out_file * ".jld2"
    if isfile(lin_stab_filename) && jld2_has_key(lin_stab_filename, "gred")
        @info "Linear stability results already exist and are compatible, skipping to next step..."
    else
        @info "Generating compatible linear stability results..."
        lin_stab_from_params(nu, gred, delta, Pr, Lz, efgm, lin_stab_out_file)
    end

    # Run DNS simulation using lin stab results and save results in dns_out_file
    KH_from_params(nu, gred, delta, Pr, Lz, Nx, Nz, enoise, T, lin_stab_out_file, dns_out_file)

    # Generate data for the combined plot only
    run_label = make_run_label(gred, delta)
    Re_vec, Ri_vec = plot_KH_b_only(dns_out_file, Lz, nu; delta=delta, run_label=run_label, save_animation=save_KH_animation, save_ri_re=false)
    if generate_Ri_animation
        plot_Ri(dns_out_file; run_label=run_label, save_animation=true, save_frame=false,
                heatmap_filename=joinpath(out_path, "Ri_heatmap_$(run_label).mp4"))
    end
    if generate_velocity_magnitude_animation
        label_filename = replace(run_label, ' ' => '_')
        plot_velocity_magnitude(dns_out_file; run_label=run_label, save_animation=true, save_frame=false,
                                heatmap_filename=joinpath(out_path, "velocity_magnitude_$(label_filename).mp4"))
    end
    if generate_dudz_animation
        label_filename = replace(run_label, ' ' => '_')
        plot_dudz(dns_out_file; run_label=run_label, save_animation=true, save_frame=false,
                  heatmap_filename=joinpath(out_path, "dudz_$(label_filename).mp4"))
    end
    if generate_velocity_tanh_fit
        label_filename = replace(run_label, ' ' => '_')
        fit_velocity_tanh_profile(dns_out_file;
                                  x_min=-50.0,
                                  x_max=50.0,
                                  prefix=joinpath(out_path, "velocity_tanh_fit_$(label_filename)"),
                                  plot_u0_filename=joinpath(out_path, "velocity_tanh_fit_$(label_filename)_u0.png"),
                                  plot_delta_filename=joinpath(out_path, "velocity_tanh_fit_$(label_filename)_delta.png"))
    end
    push!(all_Reynolds, Re_vec)
    push!(all_Rig, Ri_vec)
    push!(all_labels, run_label)
    push!(all_theory_coeffs, gred * delta^3 / nu^2)
    @info "-----------------------------------------------"
end

if generate_Ri_Re_plot && length(all_Reynolds) > 1
    plot_Ri_vs_Re(all_Reynolds, all_Rig, all_labels, all_theory_coeffs, sweep_param)
end

runs_time = (time_ns() - runs_start_time) * 1e-9

@info "-----------------------------------------------------------------------------------------------"
@info "ALL RUNS COMPLETED IN $(prettytime(runs_time))"
@info "-----------------------------------------------------------------------------------------------"