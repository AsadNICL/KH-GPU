using Plots
using JLD2
using Printf
using Oceananigans

function plot_dudz(filename; run_label="", save_animation=true, save_frame=false,
                    heatmap_filename="dudz_heatmap.mp4", frame_filename="dudz_heatmap.png",
                    plot_size=(1600, 900))
    @info "Generating du/dz heatmap animation..."
    ts_file = filename * ".jld2"

    u_ic = FieldTimeSeries(ts_file, "u", iterations = 0)
    xb, yb, zb = nodes(u_ic)
    xlims = (minimum(xb), maximum(xb))
    zlims = (minimum(zb), maximum(zb))

    file_xz = jldopen(ts_file)
    iterations = sort(parse.(Int, keys(file_xz["timeseries/t"])))

    # Determine global color limits for consistent scaling
    dudz_min = Inf
    dudz_max = -Inf
    for iter in iterations
        u_xz = file_xz["timeseries/u/$iter"][:, 1, :]
        dudz = compute_dudz(u_xz, zb)
        dudz_min = min(dudz_min, minimum(dudz))
        dudz_max = max(dudz_max, maximum(dudz))
    end

    anim = @animate for iter in iterations
        u_xz = file_xz["timeseries/u/$iter"][:, 1, :]
        t = file_xz["timeseries/t/$iter"]
        dudz = compute_dudz(u_xz, zb)

        heatmap(xb, zb, dudz'; color = :thermal,
                xlabel = "x", ylabel = "z",
                aspect_ratio = :equal,
                xlims = xlims, ylims = zlims,
                clims = (dudz_min, dudz_max),
            title = @sprintf("du/dz, t = %.2f %s", t, run_label),
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

function compute_dudz(u_xz, zb)
    nx, nz = size(u_xz)
    dudz = zeros(Float64, nx, nz)
    dz = diff(zb)

    for ix in 1:nx
        dudz[ix, 1] = (u_xz[ix, 2] - u_xz[ix, 1]) / dz[1]
        for iz in 2:nz-1
            dudz[ix, iz] = (u_xz[ix, iz+1] - u_xz[ix, iz-1]) / (zb[iz+1] - zb[iz-1])
        end
        dudz[ix, nz] = (u_xz[ix, nz] - u_xz[ix, nz-1]) / dz[end]
    end

    return dudz
end