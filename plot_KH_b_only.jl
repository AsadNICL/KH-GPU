# Reads in the JLD2 output written by KH_from_params.jl and saves a buoyancy-only animation
# (lighter alternative to plot_KH.jl which also plots log10(chi)).
function plot_KH_b_only(filename, Lz, nu; delta=1.0, run_label="", save_animation=true, save_ri_re=true, ri_re_filename="Ri_vs_Re.png")

    @info "Generating plots..."
    plot_start_time = time_ns()

    # Read in the first iteration.  We do this to load the grid
    # filename * ".jld2" concatenates the extension to the end of the filename
    b_ic = FieldTimeSeries(filename * ".jld2", "b", iterations = 0)

    # Load in coordinate arrays
    # We do this separately for each variable since Oceananigans uses a staggered grid
    xb, yb, zb = nodes(b_ic)
    xlims = (minimum(xb), maximum(xb))
    ylims = (minimum(zb), maximum(zb))

    Lx = maximum(xb) - minimum(xb)
    x_inds = findall(abs.(xb) .<= Lx / 20)
    z_inds = findall(abs.(zb) .<= delta / 2)
    z_inds = filter(i -> i > 1 && i < length(zb), z_inds)
    if isempty(x_inds) || isempty(z_inds)
        error("No grid points found in Richardson region for delta=$(delta) and Lx=$(Lx)")
    end

    # Now, open the file with our data
    file_xz = jldopen(filename * ".jld2")

    # Extract a vector of iterations
    iterations = parse.(Int, keys(file_xz["timeseries/t"]))

    # SET UP RICH GRAPH
    Rig = Float64[]
    Reynolds = Float64[]

    @info "Making an animation from saved data..."

    # First pass: find global min/max for consistent color limits
    b_global_min, b_global_max = Inf, -Inf


    for iter in iterations
        b_data = file_xz["timeseries/b/$iter"][:, 1, :]
        
        b_global_min = min(b_global_min, minimum(b_data))
        b_global_max = max(b_global_max, maximum(b_data))
    end

    times = Float64[]

    # Here, we loop over all iterations
    anim = @animate for (i, iter) in enumerate(iterations)

        # @info "Drawing frame $i from iteration $iter..."

        # Load data for this iteration
        u_xz = file_xz["timeseries/u/$iter"][:,1,:];
        w_xz = file_xz["timeseries/w/$iter"][:,1,:];
        b_xz = file_xz["timeseries/b/$iter"][:, 1, :];
        t = file_xz["timeseries/t/$iter"];

        # Compute a fitted tanh profile for the streamwise velocity and use its shear in Ri.
        # This avoids using the instantaneous, noisy finite-difference dudz in each grid cell.
        u_avg = vec(mean(u_xz[x_inds, :], dims = 1))
        fitted_u0, fitted_delta = try
            if isdefined(Main, :tanh_fit_u0_delta)
                tanh_fit_u0_delta(zb, u_avg; delta_min=1e-3, delta_max=maximum(abs.(zb)))
            else
                throw(ErrorException("tanh fitter not available"))
            end
        catch
            (0.0, NaN)
        end

        if isfinite(fitted_delta) && fitted_delta > 0
            dudz_profile = fitted_u0 / fitted_delta .* sech.(zb ./ fitted_delta) .^ 2
        else
            dudz_profile = similar(zb, Float64)
            for iz in eachindex(zb)
                if iz == 1
                    dz = zb[2] - zb[1]
                    dudz_profile[iz] = (u_xz[1, 2] - u_xz[1, 1]) / dz
                elseif iz == length(zb)
                    dz = zb[end] - zb[end-1]
                    dudz_profile[iz] = (u_xz[1, end] - u_xz[1, end-1]) / dz
                else
                    dz = zb[iz+1] - zb[iz-1]
                    dudz_profile[iz] = (u_xz[1, iz+1] - u_xz[1, iz-1]) / dz
                end
            end
        end

        # Compute Richardson over the requested region and take the RMS
        nx = length(x_inds)
        nz = length(z_inds)
        Ri_region = zeros(Float64, nx, nz)
        for (ix_idx, ix_val) in enumerate(x_inds)
            for (iz_idx, iz_val) in enumerate(z_inds)
                dz = zb[iz_val+1] - zb[iz_val-1]
                dbdz = (b_xz[ix_val, iz_val+1] - b_xz[ix_val, iz_val-1]) / dz
                dudz = dudz_profile[iz_val]
                Ri_region[ix_idx, iz_idx] = dbdz / (dudz^2 + 1e-12)
            end
        end
        Ri_rms = sqrt(mean(Ri_region .^ 2))

        # Prefer using fitted tanh amplitude `u0` for characteristic velocity when available.
        # Fallback to instantaneous Urms if the fitter is not available or fails.
        u0 = try
            if isdefined(Main, :tanh_fit_u0_delta)
                fitted_u0
            else
                throw(ErrorException("tanh fitter not available"))
            end
        catch
            sqrt(mean(u_xz.^2) + mean(w_xz.^2))
        end
        # Use `delta` (model thickness) as the characteristic length scale for Re
        Re_inst = 1.414*abs(u0) * delta / nu

        # Store values for time tracking
        push!(times, t)
        push!(Rig, Ri_rms)
        push!(Reynolds, Re_inst)

        # Create titles
        b_title = @sprintf("b, t = %.2f", t)

        
        # Use global limits for consistent visualization across all frames
        b_lims = (b_global_min, b_global_max)
        #x_lims = (minimum(xb), maximum(xb))
        x_center = (minimum(xb) + maximum(xb)) / 2
        x_lims = (x_center - 20, x_center + 20)
        z_lims = (minimum(zb), maximum(zb))
        
        # Create plots with Plots.jl syntax
        b_xz_plot = heatmap(xb, zb, b_xz', color = :viridis, xlabel = "x", ylabel = "z", aspect_ratio = :equal, xlims = x_lims, ylims = z_lims, clims = b_lims, title=b_title); 
  
        # Combine the sub-plots into a single figure (vertical layout)
        plot(b_xz_plot)
        #plot(b_xz_plot,
        #    size = (3200, 3200))

    end

    #richplot = plot(times, Rig, xlabel = "Time", ylabel = "Ri", lw = 2, legend = false, title = "Gradient Richardson number at origin", xlims = (0, maximum(times)), ylims = (0, 1), size = (800, 400))

     # Safe axis limits: avoid reducing over empty collections (can happen if values are NaN or non-positive)
     finite_Re = Reynolds[isfinite.(Reynolds)]
     pos_Re = finite_Re[finite_Re .> 0]
     x_min = isempty(pos_Re) ? (isempty(finite_Re) ? 1e-12 : minimum(abs.(finite_Re))) : minimum(pos_Re)
     x_max = isempty(finite_Re) ? 1.0 : maximum(finite_Re)

     finite_Ri = Rig[isfinite.(Rig)]
     y_min = isempty(finite_Ri) ? 0.0 : minimum(finite_Ri)
     y_max = isempty(finite_Ri) ? 1.0 : maximum(finite_Ri)

     Ri_Re_plot = plot(Reynolds, Rig,
                             xlabel = "Reynolds number",
                             ylabel = "Gradient Richardson number",
                             lw = 2,
                             legend = false,
                             title = "Ri vs Reynolds",
                             xscale = :log10,
                             yscale = :log10,
                             xlims = (x_min, x_max),
                             ylims = (y_min, y_max),
                             size = (800, 400))

    close(file_xz)

    # Save the animation to a file
    if save_animation
        mp4(anim, filename*".mp4", fps = 10)
    end

    if save_ri_re
        if ri_re_filename == "Ri_vs_Re.png" && run_label != ""
            ri_re_filename = "Ri_vs_Re_$(run_label).png"
        end
        savefig(Ri_Re_plot, ri_re_filename)
    end

    # Display plot time
    plot_time = (time_ns() - plot_start_time) * 1e-9
    @info "   Plots completed in $(prettytime(plot_time))"

    return Reynolds, Rig
end  
