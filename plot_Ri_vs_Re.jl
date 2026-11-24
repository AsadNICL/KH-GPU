using Plots
using Printf

function truncate_curve_for_positive_gradient(Re, Ri; min_points=2)
    if length(Re) < min_points
        return collect(Re), collect(Ri), false
    end

    for j in min_points:length(Re)
        start_idx = j - min_points + 1
        x_window = log10.(Re[start_idx:j])
        y_window = log10.(Ri[start_idx:j])

        if any(v -> !isfinite(v) || v <= 0, Re[start_idx:j]) ||
           any(v -> !isfinite(v) || v <= 0, Ri[start_idx:j]) ||
           any(!isfinite, x_window) || any(!isfinite, y_window)
            continue
        end

        slope = (y_window[end] - y_window[1]) / (x_window[end] - x_window[1])
        if slope > 0
            return Re[1:j], Ri[1:j], true
        end
    end

    return collect(Re), collect(Ri), false
end

function plot_Ri_vs_Re(all_Reynolds, all_Rig, all_labels, all_theory_coeffs, sweep_param; plot_filename=nothing, all_nu=nothing, all_Gr=nothing, all_initial_tilt_angles=nothing, all_Lx=nothing, all_Lz=nothing, cutoff_on_positive_gradient=false, label_by=:nu)
    plot_title = sweep_param == :delta ? "log Ri vs log Re for multiple values of delta" :
                 sweep_param == :gred ? "log Ri vs log Re for multiple values of reduced gravity" :
                 sweep_param == :nu ? "log Ri vs log Re for multiple values of viscosity" :
                 sweep_param == :initial_tilt_angle ? "log Ri vs log Re for multiple values of theta" :
                 sweep_param == :Lx ? "log Ri vs log Re for multiple values of Lx" :
                 sweep_param == :Lz ? "log Ri vs log Re for multiple values of Lz" :
                 "log Ri vs log Re"

    if plot_filename === nothing
        plot_filename = sweep_param == :delta ? "Ri_vs_Re_by_delta.png" :
                        sweep_param == :gred ? "Ri_vs_Re_by_gred.png" :
                        sweep_param == :nu ? "Ri_vs_Re_by_nu.png" :
                        sweep_param == :initial_tilt_angle ? "Ri_vs_Re_by_theta.png" :
                        sweep_param == :Lx ? "Ri_vs_Re_by_Lx.png" :
                        sweep_param == :Lz ? "Ri_vs_Re_by_Lz.png" :
                        "Ri_vs_Re.png"
    end

    ncurves = length(all_Reynolds)
    colors = palette(:auto, ncurves)

    positive_Re = filter(v -> isfinite(v) && v > 0, vcat(all_Reynolds...))
    positive_Ri = filter(v -> isfinite(v) && v > 0, vcat(all_Rig...))

    x_min = isempty(positive_Re) ? 1e2 : minimum(positive_Re)
    x_max = isempty(positive_Re) ? 1e6 : maximum(positive_Re)
    y_min = isempty(positive_Ri) ? 1e-1 : minimum(positive_Ri)
    y_max = isempty(positive_Ri) ? 1.0 : maximum(positive_Ri)

    agg_plot = plot(xlabel = "log10(Reynolds number)",
                    ylabel = "log10(Gradient Richardson number)",
                    title = plot_title,
                    legend = :outertopright,
                    seriestype = :path,
                    marker = :none,
                    size = (1400, 900),
                    xrotation = 25,
                    left_margin = 20mm,
                    bottom_margin = 20mm,
                    top_margin = 12mm,
                    right_margin = 12mm,
                    fontsize = 20,
                    color = colors,
                    guidefontsize = 15,
                    legendfontsize = 20,
                    xscale = :log10,
                    yscale = :log10)

    if label_by == :Gr && all_Gr !== nothing
        label_prefix = i -> @sprintf("Gr = %.1e", all_Gr[i])
    elseif sweep_param == :initial_tilt_angle && all_initial_tilt_angles !== nothing
        label_prefix = i -> @sprintf("θ0 = %.1f°", all_initial_tilt_angles[i])
    elseif sweep_param == :Lx && all_Lx !== nothing
        label_prefix = i -> @sprintf("Lx = %.1f", all_Lx[i])
    elseif sweep_param == :Lz && all_Lz !== nothing
        label_prefix = i -> @sprintf("Lz = %.1f", all_Lz[i])
    elseif all_nu !== nothing
        label_prefix = i -> @sprintf("ν = %.1e", all_nu[i])
    else
        label_prefix = x -> x
    end

    for i in eachindex(all_Reynolds)
        if label_by == :Gr && all_Gr !== nothing
            curve_label = label_prefix(i)
        elseif sweep_param == :initial_tilt_angle && all_initial_tilt_angles !== nothing
            curve_label = label_prefix(i)
        elseif sweep_param == :Lx && all_Lx !== nothing
            curve_label = label_prefix(i)
        elseif sweep_param == :Lz && all_Lz !== nothing
            curve_label = label_prefix(i)
        else
            curve_label = all_nu === nothing && all_Gr === nothing ? all_labels[i] : label_prefix(i)
        end

        if cutoff_on_positive_gradient
            curve_Re, curve_Ri, cutoff_flag = truncate_curve_for_positive_gradient(all_Reynolds[i], all_Rig[i])
            plot!(agg_plot, curve_Re, curve_Ri, lw = 2, seriestype = :path,
                  marker = :none, label = curve_label, color = colors[i])
            if cutoff_flag
                scatter!(agg_plot, [curve_Re[end]], [curve_Ri[end]], marker = :circle,
                         markersize = 6, color = colors[i], label = "")
            end
        else
            plot!(agg_plot, all_Reynolds[i], all_Rig[i], lw = 2, seriestype = :path,
                  marker = :none, label = curve_label, color = colors[i])
        end
    end

    theory_Re = exp10.(range(log10(x_min), stop = log10(x_max), length = 200))
    theory_indices = sweep_param == :initial_tilt_angle ? [1] : eachindex(all_theory_coeffs)
    for i in theory_indices
        theory_coeff = all_theory_coeffs[i]
        theory_Ri = theory_coeff .* theory_Re .^ (-2)
        if sweep_param == :initial_tilt_angle
            theory_label = "theory"
        elseif label_by == :Gr
            theory_label = all_Gr === nothing ? "theory " * all_labels[i] : "theory Gr = " * @sprintf("%.1e", all_Gr[i])
        else
            theory_label = all_nu === nothing ? "theory " * all_labels[i] : "theory ν = " * @sprintf("%.1e", all_nu[i])
        end
        plot!(agg_plot, theory_Re, theory_Ri, lw = 1, ls = :dash, seriestype = :path,
              marker = :none, label = theory_label, color = colors[1])
    end

    box_xmin = 1.0e5
    box_ymin = 0.15
    box_ymax = 0.22
    box_xmax = 1.0e6
    box_x = [box_xmin, box_xmin, box_xmax, box_xmax, box_xmin]
    box_y = [box_ymin, box_ymax, box_ymax, box_ymin, box_ymin]
    plot!(agg_plot, box_x, box_y, fill = (0, 0.25), linecolor = :green, color = :green,
          label = "Re > 10^5, 0.15 < Ri < 0.22", lw = 0)

    xlims!(agg_plot, (1.0e4, max(x_max, 1.0e6)))
    ylims!(agg_plot, (3.0e-2, 1.0))
    savefig(agg_plot, plot_filename)
    @info "Saved aggregate plot to $(plot_filename)"
    return agg_plot
end
