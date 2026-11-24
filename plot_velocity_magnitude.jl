using Plots
using JLD2
using Printf
using Oceananigans

# Plot the velocity magnitude as a time-evolving heatmap in x-z.
function plot_velocity_magnitude(filename; run_label="", save_animation=true, save_frame=false,
                                 heatmap_filename="velocity_magnitude.mp4", frame_filename="velocity_magnitude.png",
                                 plot_size=(1600, 900))

    @info "Generating velocity magnitude heatmap animation..."
    ts_file = filename * ".jld2"

    # Load grid coordinates from the first iteration to get x and z node positions.
    b_ic = FieldTimeSeries(ts_file, "b", iterations = 0)
    xb, yb, zb = nodes(b_ic)
    xlims = (minimum(xb), maximum(xb))
    zlims = (minimum(zb), maximum(zb))

    file_xz = jldopen(ts_file)
    iterations = parse.(Int, keys(file_xz["timeseries/t"]))

    anim = @animate for iter in iterations
        u_xz = file_xz["timeseries/u/$iter"][:, 1, :]
        w_xz = file_xz["timeseries/w/$iter"][:, 1, :]
        t = file_xz["timeseries/t/$iter"]

        # Align u and w to the same x-z locations.
        u_centered = 0.5 .* (u_xz[1:end-1, :] .+ u_xz[2:end, :])
        w_centered = 0.5 .* (w_xz[:, 1:end-1] .+ w_xz[:, 2:end])

        velocity_magnitude = sqrt.(u_centered .^ 2 .+ w_centered .^ 2)

        heatmap(xb, zb, velocity_magnitude'; color = :viridis,
            xlabel = "x", ylabel = "z",
            aspect_ratio = :equal,
            xlims = xlims, ylims = zlims,
            title = @sprintf("|u|, t = %.2f %s", t, run_label),
            size = plot_size,
            colorbar_tickfont = font(14),
            colorbar_titlefont = font(14))
    end

    close(file_xz)

    if save_animation
        mp4(anim, heatmap_filename, fps = 10)
    end

    if save_frame
        savefig(anim[end], frame_filename)
    end

    return nothing
end
