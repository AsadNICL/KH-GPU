using Plots
using Measures
using Printf

function make_ssi_plot(; output_filename = joinpath(pwd(), "Ri_vs_Re_SSI_frontier.png"))
    red_x = [
        (24300.0, 0.169),
        (30375.0, 0.169),
        (40500.0, 0.169),
        (108530.0, 0.212),
        (112871.0, 0.196),
    ]

    green_circle = [
        (1869231.0, 0.169),
        (121500.0, 0.169),
        (60750.0, 0.169),
        (117212.0, 0.182),
        (125894.0, 0.158),
        (130236.0, 0.147),
        (136748.0, 0.134),
        (138918.0, 0.130),
        (141089.0, 0.166),
        (86824.0, 0.147),
        (65118.0, 0.147),
        (52094.0, 0.147),
    ]

    yellow_square = [(48600.0, 0.169)]

    box_xmin = 1.0e5
    box_ymin = 0.15
    box_ymax = 0.22
    box_xmax = 1.0e6
    box_x = [box_xmin, box_xmin, box_xmax, box_xmax, box_xmin]
    box_y = [box_ymin, box_ymax, box_ymax, box_ymin, box_ymin]

    p = plot(
        xlabel = "log10(Reynolds number)",
        ylabel = "log10(Gradient Richardson number)",
        title = "log Ri vs log Re",
        legend = :outertopright,
        xscale = :log10,
        yscale = :log10,
        xlims = (9.0e3, 1.0e6),
        ylims = (0.1, 0.3),
        size = (1400, 900),
        markerstrokewidth = 1.5,
        grid = true,
        left_margin = 20mm,
        bottom_margin = 20mm,
        top_margin = 12mm,
        right_margin = 12mm,
        fontsize = 20,
        guidefontsize = 15,
        legendfontsize = 17,
    )

    plot!(p, box_x, box_y,
          fill = (0, 0.25),
          linecolor = :green,
          color = :green,
          lw = 0,
          label = "Re ∈ [1e5, 1e6], Ri ∈ [0.15, 0.22]")

    scatter!(p, first.(red_x), last.(red_x),
             marker = :x,
             markersize = 10,
             color = :red,
             markeralpha = 1,
             label = "No SSI")

    scatter!(p, first.(green_circle), last.(green_circle),
             marker = :circle,
             markersize = 8,
             color = :green,
             markeralpha = 1,
             label = "SSI")

    scatter!(p, first.(yellow_square), last.(yellow_square),
             marker = :square,
             markersize = 9,
             color = :yellow,
             markeralpha = 1,
             label = "borderline")

    savefig(p, output_filename)
    @info "Saved SSI diagnostic plot to $(output_filename)"
    return p
end

p = make_ssi_plot()
display(p)
