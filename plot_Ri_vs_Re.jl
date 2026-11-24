using Plots
using Printf

function plot_Ri_vs_Re(all_Reynolds, all_Rig, all_labels, all_theory_coeffs, sweep_param; plot_filename=nothing)
    plot_title = sweep_param == :delta ? "Ri vs Reynolds for multiple values of delta" :
                 sweep_param == :gred ? "Ri vs Reynolds for multiple values of reduced gravity" :
                 "Ri vs Reynolds for multiple values of gred and delta"

    if plot_filename === nothing
        plot_filename = sweep_param == :delta ? "Ri_vs_Re_by_delta.png" :
                        sweep_param == :gred ? "Ri_vs_Re_by_gred.png" :
                        "Ri_vs_Re_by_gred_delta.png"
    end

    ncurves = length(all_Reynolds)
    colors = palette(:auto, ncurves)

    x_min = minimum(vcat(all_Reynolds...))
    x_max = maximum(vcat(all_Reynolds...))
    y_min = minimum(vcat(all_Rig...))

    y_min = min(y_min, 0.0)

    agg_plot = plot(xlabel = "Reynolds number",
                    ylabel = "Gradient Richardson number",
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
                    fontsize = 15,
                    color = colors,
                    guidefontsize = 30,
                    legendfontsize = 20)

    for i in eachindex(all_Reynolds)
        plot!(agg_plot, all_Reynolds[i], all_Rig[i], lw = 2, seriestype = :path,
              marker = :none, label = all_labels[i], color = colors[i])
    end

    theory_Re = range(x_min, stop = x_max, length = 100)
    for i in eachindex(all_theory_coeffs)
        theory_coeff = all_theory_coeffs[i]
        theory_Ri = theory_coeff .* theory_Re .^ (-2)
        plot!(agg_plot, theory_Re, theory_Ri, lw = 1, ls = :dash, seriestype = :path,
              marker = :none, label = "theory " * all_labels[i], color = colors[i])
    end

    xlims!(agg_plot, (x_min, x_max))
    ylims!(agg_plot, (y_min, 1.0))
    savefig(agg_plot, plot_filename)
    @info "Saved aggregate plot to $(plot_filename)"
    return agg_plot
end
