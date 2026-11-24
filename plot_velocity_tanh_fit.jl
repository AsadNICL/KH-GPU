using Plots
using JLD2
using Printf
using Oceananigans
using LinearAlgebra
using Statistics

function tanh_fit_u0_delta(z, u_avg; delta_min=1e-3, delta_max=nothing)
    if delta_max === nothing
        delta_max = maximum(abs.(z))
    end
    delta_min = max(delta_min, eps(Float64))
    delta_max = max(delta_max, delta_min * 10)

    objective(log_delta) = begin
        delta = exp(log_delta)
        t = tanh.(z ./ delta)
        u0 = dot(u_avg, t) / dot(t, t)
        sum(abs2.(u_avg .- u0 .* t))
    end

    log_lo = log(delta_min)
    log_hi = log(delta_max)
    phi = (sqrt(5) - 1) / 2
    c = log_hi - phi * (log_hi - log_lo)
    d = log_lo + phi * (log_hi - log_lo)
    fc = objective(c)
    fd = objective(d)

    for _ in 1:40
        if fc < fd
            log_hi = d
            d = c
            fd = fc
            c = log_hi - phi * (log_hi - log_lo)
            fc = objective(c)
        else
            log_lo = c
            c = d
            fc = fd
            d = log_lo + phi * (log_hi - log_lo)
            fd = objective(d)
        end
    end

    best_log_delta = (log_lo + log_hi) / 2
    best_delta = exp(best_log_delta)
    t = tanh.(z ./ best_delta)
    best_u0 = dot(u_avg, t) / dot(t, t)
    return best_u0, best_delta
end

function fit_velocity_tanh_profile(filename;
                                    x_min=-50.0,
                                    x_max=50.0,
                                    save_plots=true,
                                    prefix="velocity_tanh_fit",
                                    plot_u0_filename = "",
                                    plot_delta_filename = "")
    ts_file = filename * ".jld2"
    u_ic = FieldTimeSeries(ts_file, "u", iterations = 0)
    xu, yu, zu = nodes(u_ic)

    x_inds = findall(x -> x_min <= x <= x_max, xu)
    if isempty(x_inds)
        @warn "Requested x range [$(x_min), $(x_max)] not found in u-grid; using full x extent."
        x_inds = collect(1:length(xu))
    end

    file_xz = jldopen(ts_file)
    iterations = sort(parse.(Int, keys(file_xz["timeseries/t"])))

    times = Float64[]
    u0_values = Float64[]
    model_delta_values = Float64[]

    for iter in iterations
        u_xz = file_xz["timeseries/u/$iter"][:, 1, :]
        t = file_xz["timeseries/t/$iter"]

        u_avg = vec(mean(u_xz[x_inds, :], dims = 1))

        u0, model_delta = tanh_fit_u0_delta(zu, u_avg; delta_min=1e-3, delta_max=maximum(abs.(zu)))

        push!(times, t)
        push!(u0_values, u0)
        push!(model_delta_values, model_delta)
    end

    close(file_xz)

    if isempty(plot_u0_filename)
        plot_u0_filename = string(prefix, "_u0_vs_time.png")
    end
    if isempty(plot_delta_filename)
        plot_delta_filename = string(prefix, "_delta_vs_time.png")
    end

    p_u0 = plot(times, u0_values,
               xlabel = "Time",
               ylabel = raw"u_0",
               title = "Fitted u_0 vs time",
               lw = 2,
               legend = false,
               size = (900, 500))

    p_delta = plot(times, model_delta_values,
                   xlabel = "Time",
                   ylabel = raw"model\_delta",
                   title = "Fitted model_delta vs time",
                   lw = 2,
                   legend = false,
                   size = (900, 500))

    if save_plots
        savefig(p_u0, plot_u0_filename)
        savefig(p_delta, plot_delta_filename)
    end

    return times, u0_values, model_delta_values, p_u0, p_delta
end