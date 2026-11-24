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
include("./animate_velocity_tanh_fit.jl")
include("./plot_dudz.jl")
include("./analyse_low_buoyancy_front.jl")

# Toggle whether to show only the centered x-window in the buoyancy animation (x ≈ [-20, 20])
# or the full domain. Set to false for the full domain and true for the historical narrowed view.
const centered_buoyancy_x_window = true
# Toggle whether to visualize the pointwise Ri heatmap animation during each run.
const generate_Ri_animation = false
# Toggle whether to visualize the velocity magnitude heatmap animation during each run.
const generate_velocity_magnitude_animation = false
# Toggle whether to fit and plot the velocity tanh profile during each run.
const generate_velocity_tanh_fit = false
# Toggle whether to animate the raw u(z) and fitted tanh profile over time during each run.
const generate_velocity_tanh_animation = false
# Toggle whether to visualize du/dz heatmap animation during each run.
const generate_dudz_animation = false
# Toggle whether to generate the aggregate Ri vs Re plot after runs.
const generate_Ri_Re_plot = true
# Choose the curve label style for the aggregate Ri vs Re plot when viscosity varies.
# Use :nu to label by viscosity or :Gr to label by Grashof number, where Gr = gred * delta^3 / nu^2.
const ri_vs_re_label_by = :Gr
# Toggle whether to stop plotting each Ri vs Re curve once the last five-point gradient becomes positive.
const cutoff_on_positive_gradient = false
# toggle whether to save the KH animation during each run (this is a large file, so only do it if you want to see the animation)
const save_KH_animation = false

# Toggle whether to compute and save the low-buoyancy front propagation analysis during each run.
const generate_low_buoyancy_front_analysis = false
# Threshold for the low-buoyancy front detection in the upper layer.
const low_buoyancy_front_threshold = -0.5
# Number of x-points to the left that must also be above threshold to suppress anomalies.
const low_buoyancy_front_anomaly_window = 2
# Vertical layer used to define the low-buoyancy front in the upper layer.
const low_buoyancy_front_z_min = 2.0
const low_buoyancy_front_z_max = 5.0

# Fixed fallback for runs without a reset time; otherwise the fit starts at
# fit_start_time_rotation_multiplier * rotation_time.
const fallback_fit_start_time = 25.0
const fit_start_time_rotation_multiplier = 2.0

function fit_start_time_for_run(rotation_time; rotate_back_to_horizontal=true, fallback=fallback_fit_start_time)
    return rotate_back_to_horizontal ? fit_start_time_rotation_multiplier * rotation_time : fallback
end

# Read parameter arrays
include("./params.jl")
params = Dict(
    :nu => nu_values,
    :gred => gred_values,
    :delta => delta_values,
    :Pr => Pr_values,
    :Lz => Lz_values,
    :Lx => Lx_values,
    :Nx => Nx_values,
    :Nz => Nz_values,
    :efgm => efgm_values,
    :enoise => enoise_values,
    :T => T_values,
    :initial_tilt_angle => initial_tilt_angle_values,
    :rotate_back_to_horizontal => rotate_back_to_horizontal_values,
    :angle_indep_rot_param => angle_indep_rot_param_values,
)

# Choose the parameter to compare in the aggregate Ri vs Re plot.
function infer_sweep_param(params)
    varying = filter(k -> length(params[k]) > 1, (:nu, :gred, :delta, :initial_tilt_angle, :Lx, :Lz))
    if :initial_tilt_angle in varying
        return :initial_tilt_angle
    elseif :Lx in varying
        return :Lx
    elseif :Lz in varying
        return :Lz
    elseif length(varying) == 1
        return varying[1]
    elseif length(varying) == 2
        @warn "Multiple parameters vary; using the first varying one in the label: $(varying[1])."
        return varying[1]
    else
        return :nu
    end
end

function make_run_label(gred, delta; nu=nothing, initial_tilt_angle=nothing, Lx=nothing, Lz=nothing)
    if sweep_param == :initial_tilt_angle && initial_tilt_angle !== nothing
        return @sprintf("theta0=%.1f", initial_tilt_angle)
    elseif sweep_param == :Lx && Lx !== nothing
        return @sprintf("Lx=%.1f", Lx)
    elseif sweep_param == :Lz && Lz !== nothing
        return @sprintf("Lz=%.1f", Lz)
    elseif sweep_param == :nu && nu !== nothing
        return @sprintf("nu=%.1e", nu)
    elseif sweep_param == :gred
        return @sprintf("gred=%.3f", gred)
    elseif sweep_param == :delta
        return @sprintf("delta=%.3f", delta)
    else
        return @sprintf("gred=%.3f, delta=%.3f", gred, delta)
    end
end

function make_ri_re_filename(gred, delta; nu=nothing, initial_tilt_angle=nothing, Lx=nothing, Lz=nothing)
    if sweep_param == :initial_tilt_angle && initial_tilt_angle !== nothing
        return @sprintf("Ri_vs_Re_theta0=%.1f.png", initial_tilt_angle)
    elseif sweep_param == :Lx && Lx !== nothing
        return @sprintf("Ri_vs_Re_Lx=%.1f.png", Lx)
    elseif sweep_param == :Lz && Lz !== nothing
        return @sprintf("Ri_vs_Re_Lz=%.1f.png", Lz)
    elseif sweep_param == :nu && nu !== nothing
        return @sprintf("Ri_vs_Re_nu=%.1e.png", nu)
    elseif sweep_param == :gred
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

# Set a shared output directory for per-run and aggregate plots.
base_path = pwd()
out_path = joinpath(base_path, "out")
mkpath(out_path)

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
all_nu = Float64[]
all_initial_tilt_angles = Float64[]
all_Lx = Float64[]
all_Lz = Float64[]
all_Gr = Float64[]
all_front_times = Vector{Vector{Float64}}()
all_front_distances = Vector{Vector{Float64}}()
all_front_labels = String[]
all_front_fit_starts = Float64[]
for irun in 1:nruns
    # Extract parameters for this run 
    @info "STARTING RUN $irun of $nruns ..."
    nu = getval(params[:nu], irun)
    gred = getval(params[:gred], irun)
    delta = getval(params[:delta], irun)
    Pr = getval(params[:Pr], irun)
    Lz = getval(params[:Lz], irun)
    Lx = getval(params[:Lx], irun)
    Nx = getval(params[:Nx], irun)
    Nz = getval(params[:Nz], irun)
    efgm = getval(params[:efgm], irun)
    enoise = getval(params[:enoise], irun)
    T = getval(params[:T], irun)
    initial_tilt_angle = getval(params[:initial_tilt_angle], irun)
    angle_indep_rot_param = getval(params[:angle_indep_rot_param], irun)
    rotate_back_to_horizontal = getval(params[:rotate_back_to_horizontal], irun)

    sin_theta = sin(deg2rad(initial_tilt_angle))
    if abs(sin_theta) < 1e-12
        error("initial_tilt_angle = $(initial_tilt_angle)° gives sin(theta0) ≈ 0, so angle_indep_rot_param cannot be converted to a reset time.")
    end
    rotation_time = angle_indep_rot_param / sin_theta
    fit_start = fit_start_time_for_run(rotation_time; rotate_back_to_horizontal=rotate_back_to_horizontal)

    # Set output paths
    today_str = Dates.format(today(), "yyyy-mm-dd")
    today_str = replace(today_str, "-" => "") # to match the required format
    lin_stab_out_name = @sprintf("KH_lin_stab_nu=%.1e_gred=%0.2f_delta=%0.2f_Pr=%d_Lz=%d_Lx=%d_efgm=%.1e",
        nu, gred, delta, Pr, Lz, Lx, efgm)
    lin_stab_out_file = joinpath(out_path, lin_stab_out_name)
    rotation_suffix = rotate_back_to_horizontal ? @sprintf("_rotation_time=%.2f", rotation_time) : ""
    dns_out_name = @sprintf("KH_%s_nu=%.1e_gred=%0.2f_delta=%0.2f_Pr=%d_Lz=%d_Lx=%d_Nx=%d_Nz=%d_T=%d_theta0=%.2f%s",
        today_str, nu, gred, delta, Pr, Lz, Lx, Nx, Nz, T, initial_tilt_angle, rotation_suffix)
    dns_out_file = joinpath(out_path, dns_out_name)

    lin_stab_filename = lin_stab_out_file * ".jld2"
    if isfile(lin_stab_filename) && jld2_has_key(lin_stab_filename, "gred")
        @info "Linear stability results already exist and are compatible, skipping to next step..."
    else
        @info "Generating compatible linear stability results..."
        lin_stab_from_params(nu, gred, delta, Pr, Lz, efgm, lin_stab_out_file)
    end

    # Run DNS simulation using lin stab results and save results in dns_out_file
    KH_from_params(nu, gred, delta, Pr, Lz, Lx, Nx, Nz, enoise, T, lin_stab_out_file, dns_out_file;
        initial_tilt_angle=initial_tilt_angle,
        rotate_back_to_horizontal=rotate_back_to_horizontal,
        rotation_time=rotation_time)

    # Generate data for the combined plot only
    run_label = make_run_label(gred, delta; nu=nu, initial_tilt_angle=initial_tilt_angle, Lx=Lx, Lz=Lz)
    Re_vec, Ri_vec = plot_KH_b_only(dns_out_file, Lz, nu;
        delta=delta,
        run_label=run_label,
        save_animation=save_KH_animation,
        save_ri_re=false,
        centered_x_window=centered_buoyancy_x_window)
    if generate_low_buoyancy_front_analysis
        front_label = @sprintf("Lx=%.1f_theta0=%.1f", Lx, initial_tilt_angle)
        if rotate_back_to_horizontal
            front_label = @sprintf("Lx=%.1f_theta0=%.1f_rot=%.1f", Lx, initial_tilt_angle, rotation_time)
        end
        front_times, front_distances, _ = analyse_low_buoyancy_front(dns_out_file;
            z_min=low_buoyancy_front_z_min,
            z_max=low_buoyancy_front_z_max,
            threshold=low_buoyancy_front_threshold,
            anomaly_window=low_buoyancy_front_anomaly_window,
            output_filename=joinpath(out_path, "low_buoyancy_front_$(front_label).png"),
            title = @sprintf("Low-buoyancy front distance, Lx=%.1f, θ0=%.1f%s", Lx, initial_tilt_angle, rotate_back_to_horizontal ? @sprintf(", rotation_time=%.1f", rotation_time) : ""),
            theta0 = initial_tilt_angle,
            rotation_time = rotation_time,
            fit_start_time = fit_start)
        push!(all_front_times, front_times)
        push!(all_front_distances, front_distances)
        push!(all_front_fit_starts, fit_start)
        fit_label = if rotate_back_to_horizontal
            @sprintf("θ0=%.1f°, rot=%.1f | ", initial_tilt_angle, rotation_time)
        else
            @sprintf("θ0=%.1f° | ", initial_tilt_angle)
        end
        slope, intercept, _, _ = fit_distance_vs_time(front_times, front_distances; t_min=fit_start)
        if slope !== nothing
            fit_label *= @sprintf("grad=%.4f, int=%.4f", slope, intercept)
        else
            fit_label *= "grad=n/a, int=n/a"
        end
        push!(all_front_labels, fit_label)
    end
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
    if generate_velocity_tanh_animation
        label_filename = replace(run_label, ' ' => '_')
        animate_velocity_tanh_profile(dns_out_file;
                                     run_label=run_label,
                                     output_filename=joinpath(out_path, "velocity_tanh_profile_$(label_filename).gif"))
    end
    push!(all_Reynolds, Re_vec)
    push!(all_Rig, Ri_vec)
    push!(all_labels, run_label)
    gr =  gred * delta^3 / nu^2
    push!(all_theory_coeffs,  2*gr)
    push!(all_nu, nu)
    push!(all_Gr, gr)
    push!(all_initial_tilt_angles, initial_tilt_angle)
    push!(all_Lx, Lx)
    push!(all_Lz, Lz)
    @info "-----------------------------------------------"
end

if generate_low_buoyancy_front_analysis && (length(unique(initial_tilt_angle_values)) > 1 || length(unique(angle_indep_rot_param_values)) > 1) && length(all_front_times) > 1
    compare_title = "Low-buoyancy front distance comparison"
    if length(unique(initial_tilt_angle_values)) > 1 && length(unique(angle_indep_rot_param_values)) > 1
        compare_title = "Low-buoyancy front distance comparison, θ0 and rotation_time"
    elseif length(unique(initial_tilt_angle_values)) > 1
        compare_title = "Low-buoyancy front distance comparison, θ0"
    elseif length(unique(angle_indep_rot_param_values)) > 1
        compare_title = "Low-buoyancy front distance comparison, rotation_time"
    end
    plot_front_distance_overlay(all_front_times, all_front_distances, all_front_labels, all_front_fit_starts;
        output_filename = joinpath(out_path, "low_buoyancy_front_comparison.png"),
        title = compare_title)
end

if generate_Ri_Re_plot && length(all_Reynolds) > 1
    plot_Ri_vs_Re(all_Reynolds, all_Rig, all_labels, all_theory_coeffs, sweep_param;
        all_nu=all_nu,
        all_Gr=all_Gr,
        all_initial_tilt_angles=all_initial_tilt_angles,
        all_Lx=all_Lx,
        all_Lz=all_Lz,
        cutoff_on_positive_gradient=cutoff_on_positive_gradient,
        label_by=ri_vs_re_label_by)
end

runs_time = (time_ns() - runs_start_time) * 1e-9

@info "-----------------------------------------------------------------------------------------------"
@info "ALL RUNS COMPLETED IN $(prettytime(runs_time))"
@info "-----------------------------------------------------------------------------------------------"