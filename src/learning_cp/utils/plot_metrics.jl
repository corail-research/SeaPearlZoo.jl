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
            plot!(figs[i], x, y, label = string(heuristic.heuristic_type[1]))  
            title!(figs[i], string("Figure ", i))
        end
    end

    for fig in figs
        display(fig)
    end
end