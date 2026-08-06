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
    Re_vec, Ri_vec = plot_KH_b_only(dns_out_file, Lz, nu; delta=delta, run_label=run_label, save_animation=false, save_ri_re=false)
    push!(all_Reynolds, Re_vec)
    push!(all_Rig, Ri_vec)
    push!(all_labels, run_label)
    push!(all_theory_coeffs, gred * delta^3 / nu^2)
    @info "-----------------------------------------------"
end

if length(all_Reynolds) > 1
    plot_title = sweep_param == :delta ? "Ri vs Reynolds for multiple values of delta" : sweep_param == :gred ? "Ri vs Reynolds for multiple values of reduced gravity" : "Ri vs Reynolds for multiple values of gred and delta"
    plot_filename = sweep_param == :delta ? "Ri_vs_Re_by_delta.png" : sweep_param == :gred ? "Ri_vs_Re_by_gred.png" : "Ri_vs_Re_by_gred_delta.png"
    @info "Generating aggregate Ri vs Reynolds plot across $(sweep_param) values..."
    ncurves = length(all_Reynolds)
    colors = palette(:auto, ncurves)

    x_min = minimum(filter(>(0), vcat(all_Reynolds...)))
    x_max = maximum(vcat(all_Reynolds...))
    y_min = minimum(filter(>(0), vcat(all_Rig...)))
    y_max = maximum(vcat(all_Rig...))

    agg_plot = plot(xlabel = "Reynolds number", ylabel = "Gradient Richardson number", title = plot_title, legend = :outertopright, seriestype = :path, marker = :none, size = (1400, 900), xrotation = 25, left_margin = 20mm, bottom_margin = 20mm, top_margin = 12mm, right_margin = 12mm, fontsize = 20, color = colors, guidefontsize = 15, legendfontsize = 12)
    for i in eachindex(all_Reynolds)
        plot!(agg_plot, all_Reynolds[i], all_Rig[i], lw = 2, seriestype = :path, marker = :none, label = all_labels[i], color = colors[i])
    end
    theory_Re = range(x_min, stop=x_max, length=100)
    for i in eachindex(all_theory_coeffs)
        theory_coeff = all_theory_coeffs[i]
        theory_Ri = theory_coeff .* theory_Re .^ (-2)
        plot!(agg_plot, theory_Re, theory_Ri, lw = 1, ls = :dash, seriestype = :path, marker = :none, label = "theory " * all_labels[i], color = colors[i])
    end
    xlims!(agg_plot, (x_min, x_max))
    ylims!(agg_plot, (y_min, 1.0))
    savefig(agg_plot, plot_filename)
    @info "Saved aggregate plot to $(plot_filename)"
end

runs_time = (time_ns() - runs_start_time) * 1e-9

@info "-----------------------------------------------------------------------------------------------"
@info "ALL RUNS COMPLETED IN $(prettytime(runs_time))"
@info "-----------------------------------------------------------------------------------------------"