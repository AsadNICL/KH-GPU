using Plots
using JLD2
using Printf
using Statistics

# Simple separable moving-average filter that ignores NaNs.
function moving_average_separable(A::AbstractMatrix{T}, kx::Int=5, kz::Int=3) where {T<:Real}
    padx = div(kx, 2)
    padz = div(kz, 2)
    nx, nz = size(A)

    # average along x (columns of each row)
    B = Array{Float64}(undef, nx, nz)
    for j in 1:nz
        col = A[:, j]
        vals = collect(col)
        finite_mask = isfinite.(vals)
        vals2 = map(x -> isfinite(x) ? x : 0.0, vals)
        cs = cumsum([0.0; vals2])
        cm = cumsum([0; Int.(finite_mask)])
        for i in 1:nx
            i1 = max(1, i - padx)
            i2 = min(nx, i + padx)
            s = cs[i2+1] - cs[i1]
            cnt = cm[i2+1] - cm[i1]
            B[i, j] = cnt > 0 ? s / cnt : NaN
        end
    end

    # average along z (rows across depth)
    C = Array{Float64}(undef, nx, nz)
    for i in 1:nx
        row = B[i, :]
        vals = collect(row)
        finite_mask = isfinite.(vals)
        vals2 = map(x -> isfinite(x) ? x : 0.0, vals)
        cs = cumsum([0.0; vals2])
        cm = cumsum([0; Int.(finite_mask)])
        for j in 1:nz
            j1 = max(1, j - padz)
            j2 = min(nz, j + padz)
            s = cs[j2+1] - cs[j1]
            cnt = cm[j2+1] - cm[j1]
            C[i, j] = cnt > 0 ? s / cnt : NaN
        end
    end

    return C
end

# Plot the Richardson number as a time-evolving heatmap in x-z.
function plot_Ri(filename; run_label="", save_animation=true, save_frame=false,
                 heatmap_filename="Ri_heatmap.mp4", frame_filename="Ri_heatmap.png",
                 plot_size=(1600, 900))

    @info "Generating Richardson number heatmap animation..."
    ts_file = filename * ".jld2"

    # Load grid coordinates from the first iteration.
    b_ic = FieldTimeSeries(ts_file, "b", iterations = 0)
    xb, yb, zb = nodes(b_ic)
    xlims = (minimum(xb), maximum(xb))
    zlims = (minimum(zb), maximum(zb))

    file_xz = jldopen(ts_file)
    iterations = parse.(Int, keys(file_xz["timeseries/t"]))

    anim = @animate for iter in iterations
        u_xz = file_xz["timeseries/u/$iter"][:, 1, :]
        b_xz = file_xz["timeseries/b/$iter"][:, 1, :]
        t = file_xz["timeseries/t/$iter"]

        Nz = size(b_xz, 2)
        nx = size(b_xz, 1)
        Ri = fill(NaN, size(b_xz))

        # u_xz is typically staggered in x and has one extra x point relative to b_xz.
        # Average u to the b grid's x locations before computing vertical gradients.
        u_centered = 0.5 .* (u_xz[1:end-1, :] .+ u_xz[2:end, :])

        for iz in 2:(Nz - 1)
            dz = zb[iz + 1] - zb[iz - 1]
            dbdz = (b_xz[:, iz + 1] - b_xz[:, iz - 1]) ./ dz
            dudz = (u_centered[:, iz + 1] - u_centered[:, iz - 1]) ./ dz
            Ri[:, iz] .= dbdz ./ (dudz .^ 2 .+ 1e-12)
        end

        # Cap Richardson number to physical range [0, 1]
        Ri .= clamp.(Ri, 0.0, 1.0)

        # Smooth Ri directly for visualization (no mean subtraction)
        Ri_vis = moving_average_separable(Ri, 7, 3)

        # Richardson is clamped to [0,1]; use that as color limits for the heatmap
        clims = (0.0, 1.0)

        heatmap(xb, zb, Ri_vis'; color = :viridis,
            xlabel = "x", ylabel = "z",
            aspect_ratio = :equal,
            xlims = xlims, ylims = zlims,
            clims = clims,
            title = @sprintf("Ri, t = %.2f %s", t, run_label),
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
