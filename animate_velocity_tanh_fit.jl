using Plots
using JLD2
using Printf
using Oceananigans
using Statistics

include("plot_velocity_tanh_fit.jl")

function animate_velocity_tanh_profile(filename; run_label="", output_filename=nothing, fps=10, yflip=true)
    ts_file = filename * ".jld2"
    if !isfile(ts_file)
        error("Simulation file not found: $(ts_file)")
    end

    b_ic = FieldTimeSeries(ts_file, "b", iterations = 0)
    xb, yb, zb = nodes(b_ic)
    z_vals = collect(zb)

    file_xz = jldopen(ts_file)
    iterations = sort(parse.(Int, keys(file_xz["timeseries/t"])))

    if output_filename === nothing
        output_filename = splitdir(filename)[2] * "_velocity_tanh_profile.gif"
    end

    anim = @animate for iter in iterations
        u_xz = file_xz["timeseries/u/$iter"][:, 1, :]
        u_avg = vec(mean(u_xz, dims = 1))
        t = file_xz["timeseries/t/$iter"]

        u0, delta = tanh_fit_u0_delta(z_vals, u_avg; delta_min=1e-3, delta_max=maximum(abs.(z_vals)))
        u_fit = u0 .* tanh.(z_vals ./ delta)

        u_min = minimum([minimum(u_avg), minimum(u_fit)])
        u_max = maximum([maximum(u_avg), maximum(u_fit)])
        u_pad = 0.1 * max(abs(u_min), abs(u_max), 1e-12)

        p = plot(u_avg, z_vals,
                 label = "raw horizontal mean u(z)",
                 lw = 2,
                 color = :blue,
                 xlabel = "velocity",
                 ylabel = "z",
                 title = @sprintf("u(z) and tanh fit, t = %.2f %s", t, run_label),
                 xlims = (u_min - u_pad, u_max + u_pad),
                 ylim = (minimum(z_vals), maximum(z_vals)),
                 yflip = yflip,
                 legend = :topright,
                 size = (1200, 800))

        plot!(p, u_fit, z_vals,
              label = @sprintf("tanh fit: u0 = %.3e, delta = %.3e", u0, delta),
              lw = 2,
              color = :red)
    end

    close(file_xz)
    gif(anim, output_filename, fps = fps)
    @info "Saved animation to $(output_filename)"
    return output_filename
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 0
        error("Usage: julia --project=. animate_velocity_tanh_fit.jl path/to/output_file_without_extension")
    end
    filename = ARGS[1]
    output_filename = length(ARGS) > 1 ? ARGS[2] : nothing
    animate_velocity_tanh_profile(filename; output_filename=output_filename)
end
