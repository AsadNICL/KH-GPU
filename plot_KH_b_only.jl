# Reads in the JLD2 output written by KH_from_params.jl and saves a buoyancy-only animation
# (lighter alternative to plot_KH.jl which also plots log10(chi)).
function plot_KH_b_only(filename,Lz,Re)

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

    # NEED THIS TO FIND GRAD RICHARDSON NO AT ORIGIN
    ix = argmin(abs.(xb .- 0.0))
    iz = argmin(abs.(zb .- 0.0))


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
        # 2 LINES BELOW ARE NEW
        u_xz = file_xz["timeseries/u/$iter"][:,1,:];
        w_xz = file_xz["timeseries/w/$iter"][:,1,:];
        b_xz = file_xz["timeseries/b/$iter"][:, 1, :];
        t = file_xz["timeseries/t/$iter"];
        #NEXT FIVE LINES ARE NEW
        dz = zb[iz+1] - zb[iz-1];
        dbdz = (b_xz[ix, iz+1] - b_xz[ix, iz-1]) / dz;
        dudz = (u_xz[ix, iz+1] - u_xz[ix, iz-1]) / dz;
        Urms = sqrt(mean(u_xz.^2) + mean(w_xz.^2))
        Re_inst = Urms * Lz * Re

        # Store values for time tracking
        push!(times, t)
        # NEW LINE BELOW
        push!(Rig, dbdz / (dudz^2 + 1e-12))
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
    Ri_Re_plot = plot(Reynolds, Rig, xlabel = "Reynolds number", ylabel = "Gradient Richardson number", lw = 2, legend = false, title = "Ri vs Re", xlims = (0, maximum(Reynolds)), ylims = (0, 1), size = (800, 400))

    close(file_xz)

    # Save the animation to a file
    mp4(anim, filename*".mp4", fps = 10) 
    #savefig(richplot, "Ri_evolution.png")
    savefig(Ri_Re_plot, "Ri_vs_Re.png")

    # Display plot time
    plot_time = (time_ns() - plot_start_time) * 1e-9
    @info "   Plots completed in $(prettytime(plot_time))"
end  
