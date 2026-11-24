using JLD2
using Plots
using Oceananigans
using Printf

# Compute the mean buoyancy across the layer z ∈ [z_min, z_max] for each x-location,
# ignoring clearly anomalous values above 1 before averaging.
function mean_buoyancy_between_z(b_xz, z_vals; z_min=3.0, z_max=5.0, max_buoyancy=1.0)
    z_mask = (z_vals .>= z_min) .& (z_vals .<= z_max)
    if !any(z_mask)
        error("Requested z-range [$z_min, $z_max] is not present in the grid.")
    end

    layer_vals = b_xz[:, z_mask]
    filtered = copy(layer_vals)
    filtered[filtered .> max_buoyancy] .= NaN
    return vec(mean(filtered; dims = 2))
end

# Locate the low-buoyancy front in the upper layer.
#
# The front is defined as the x-position where the depth-averaged buoyancy in
# z ∈ [3, 5] changes from below the buoyancy threshold to above it while moving
# leftward from the right-hand wall. To reject isolated anomalies, the previous
# `anomaly_window` points immediately to the left of the front are also required
# to be above threshold.
function find_low_buoyancy_front(x_vals, depth_mean_b; threshold=0.3, anomaly_window=8)
    n = length(x_vals)
    front_idx = nothing

    # Scan from the right wall toward the left. The front is the first value above
    # threshold after a below-threshold-to-above-threshold transition, and the
    # previous `anomaly_window` points must also be above threshold to suppress local anomalies.
    for i in reverse(anomaly_window+2:n-1)
        if depth_mean_b[i] > threshold &&
           all(depth_mean_b[max(1, i-j)] > threshold for j in 1:anomaly_window) &&
           depth_mean_b[i+1] <= threshold
            front_idx = i
            break
        end
    end

    # Fallback: if no strict transition is found, use the rightmost positive
    # region above threshold that is robust to local anomalies.
    if front_idx === nothing
        for i in reverse(anomaly_window+2:n-1)
            if depth_mean_b[i] > threshold &&
               all(depth_mean_b[max(1, i-j)] > threshold for j in 1:anomaly_window)
                front_idx = i
                break
            end
        end
    end

    if front_idx === nothing
        return NaN
    end

    return x_vals[front_idx]
end

# Fit a straight line to distance-vs-time data for times >= t_min using a least-squares fit.
function fit_distance_vs_time(times::AbstractVector, distances::AbstractVector; t_min=25.0)
    valid = times .>= t_min
    if count(valid) < 2
        return nothing, nothing, nothing, nothing
    end

    x_fit = times[valid]
    y_fit = distances[valid]
    A = hcat(ones(length(x_fit)), x_fit)
    coeffs = A \ y_fit
    slope = coeffs[2]
    intercept = coeffs[1]
    fit_x = [minimum(x_fit), maximum(x_fit)]
    fit_y = slope .* fit_x .+ intercept
    return slope, intercept, fit_x, fit_y
end

# Plot an overlaid set of front-distance curves, with each series labelled by its
# parameter values and the best-fit line coefficients for data above its own fit start time.
function plot_front_distance_overlay(all_times, all_distances, all_labels, all_fit_starts;
    output_filename = "low_buoyancy_front_overlay.png",
    title = "Low-buoyancy front distance")

    p = plot(; xlabel = "Time",
             ylabel = "Distance from right wall",
             title = title,
             lw = 2,
             legend = :topleft,
             size = (1000, 550),
             xlims = (5.0, nothing),
             ylims = (0.0, 120.0),
             left_margin = 12mm,
             bottom_margin = 10mm,
             top_margin = 8mm,
             right_margin = 8mm,
             guidefontsize = 16,
             tickfontsize = 12,
             legendfontsize = 14)

    for i in eachindex(all_times)
        times = all_times[i]
        distances = all_distances[i]
        label = all_labels[i]
        fit_start = all_fit_starts[i]

        plot!(p, times, distances,
              label = nothing,
              marker = :circle,
              markersize = 2,
              color = i,
              alpha = 0.7)

        slope, intercept, fit_x, fit_y = fit_distance_vs_time(times, distances; t_min=fit_start)
        if slope !== nothing
            plot!(p, fit_x, fit_y,
                  label = label,
                  lw = 3,
                  ls = :dash,
                  color = i)
        end
    end

    savefig(p, output_filename)
    @info "Saved combined front-distance plot to $(output_filename)"
    return p
end

# Measure how far the low-buoyancy front is from the right wall as a function of time.
function analyse_low_buoyancy_front(filename;
    z_min = 3.0,
    z_max = 5.0,
    threshold = 0.3,
    anomaly_window = 8,
    output_filename = "",
    save_plot = true,
    title = "Low-buoyancy front distance",
    theta0 = nothing,
    rotation_time = nothing,
    fit_start_time = 25.0)

    ts_file = filename * ".jld2"
    if !isfile(ts_file)
        error("No JLD2 file found for $(filename). Did you pass the run name without the .jld2 extension?")
    end

    b_ic = FieldTimeSeries(ts_file, "b", iterations = 0)
    xb, yb, zb = nodes(b_ic)
    x_vals = xb
    right_wall_x = maximum(x_vals)

    file_xz = jldopen(ts_file)
    iterations = sort(parse.(Int, keys(file_xz["timeseries/t"])))

    times = Float64[]
    distances = Float64[]

    for iter in iterations
        b_xz = file_xz["timeseries/b/$iter"][:, 1, :]
        depth_mean_b = mean_buoyancy_between_z(b_xz, zb; z_min=z_min, z_max=z_max, max_buoyancy=1.0)
        front_x = find_low_buoyancy_front(x_vals, depth_mean_b; threshold=threshold, anomaly_window=anomaly_window)

        if isfinite(front_x)
            push!(times, file_xz["timeseries/t/$iter"])
            push!(distances, right_wall_x - front_x)
        end
    end

    close(file_xz)

    fit_mask = times .>= fit_start_time
    p = plot(times, distances,
             xlabel = "Time",
             ylabel = "Distance from right wall",
             title = title,
             lw = 2,
             label = "Measured distance",
             legend = :topleft,
             marker = :circle,
             markersize = 2,
             size = (1000, 550),
             xlims = (5.0, nothing),
             ylims = (0.0, 120.0),
             left_margin = 12mm,
             bottom_margin = 10mm,
             top_margin = 8mm,
             right_margin = 8mm,
             guidefontsize = 16,
             tickfontsize = 12,
             legendfontsize = 14)

    slope, intercept, fit_x, fit_y = fit_distance_vs_time(times, distances; t_min=fit_start_time)
    if slope !== nothing
        fit_label = if theta0 !== nothing && rotation_time !== nothing
            @sprintf("θ0=%.1f°, rot=%.1f | grad=%.4f, int=%.4f", theta0, rotation_time, slope, intercept)
        elseif theta0 !== nothing
            @sprintf("θ0=%.1f° | grad=%.4f, int=%.4f", theta0, slope, intercept)
        elseif rotation_time !== nothing
            @sprintf("rot=%.1f | grad=%.4f, int=%.4f", rotation_time, slope, intercept)
        else
            @sprintf("Best fit: grad=%.4f, int=%.4f", slope, intercept)
        end
        plot!(p, fit_x, fit_y,
              label = fit_label,
              lw = 3,
              ls = :dash,
              color = :black)
    end

    if save_plot
        if isempty(output_filename)
            output_filename = string(filename, "_low_buoyancy_front.png")
        end
        savefig(p, output_filename)
        @info "Saved front-distance plot to $(output_filename)"
    end

    return times, distances, p
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        error("Usage: julia --project=. analyse_low_buoyancy_front.jl path/to/run_without_extension [output.png]")
    end

    filename = ARGS[1]
    output_filename = length(ARGS) >= 2 ? ARGS[2] : ""

    analyse_low_buoyancy_front(filename; output_filename=output_filename)
end
