using Plots
using CSV
using DataFrames


function plot_first_solution(csv_path::AbstractString)
    df = CSV.read(csv_path, DataFrame)
    instance_ids = unique(df.num_instance)

    figs = [plot() for _ in instance_ids]

    for (i,inst) in enumerate(instance_ids)
        instance = df[df.num_instance .== inst, :]

        heuristics = groupby(instance, :num_heuristic)

        for heuristic in heuristics
            heuristic = sort(heuristic, :num_experiment)
            x = heuristic.num_experiment
            y = heuristic.first_sol
            if heuristic.heuristic_type[1] == "SimpleLearnedHeuristic"
                plot!(figs[i], x, -y, label = string(heuristic.heuristic_type[1]), linewidth=5)
            else
                plot!(figs[i], x, -y, label = string(heuristic.heuristic_type[1]))
            end
            title!(figs[i], string("Instance", i))
        end
    end

    for fig in figs
        display(fig)
    end
end

function plot_first_solution(df::DataFrames.DataFrame)
    instance_ids = unique(df.num_instance)

    figs = [plot(size=(1200, 1600)) for _ in instance_ids]

    for (i, inst) in enumerate(instance_ids)
        instance = df[df.num_instance.==inst, :]

        heuristics = groupby(instance, :num_heuristic)

        for heuristic in heuristics
            heuristic = sort(heuristic, :num_experiment)
            x = heuristic.num_experiment
            y = Float64.(heuristic.first_sol) # Convert metric value to Float64
            if heuristic.heuristic_type[1] == "SimpleLearnedHeuristic"
                plot!(figs[i], x, -y, label=string(heuristic.heuristic_type[1]), linewidth=5)
            else
                plot!(figs[i], x, -y, label=string(heuristic.heuristic_type[1]))
            end
            xlabel!(figs[i], "Evaluation step")
            ylabel!(figs[i], "Optimum")
            title!(figs[i], string("Instance", i))
        end
    end

    num_plots = length(figs)
    num_rows = div(num_plots, 2) + mod(num_plots, 2)
    plot_grid = plot(figs..., layout=(num_rows, 2), legend=:topleft)

    display(plot_grid)
end