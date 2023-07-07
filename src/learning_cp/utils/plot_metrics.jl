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
            title!(figs[i], string("First objective value for instance ", i))
        end
    end

    for fig in figs
        display(fig)
    end
end

function plot_last_solution(csv_path::AbstractString)
    df = CSV.read(csv_path, DataFrame)


    instance_ids = unique(df.num_instance)

    figs = [plot() for _ in instance_ids]

    for (i,inst) in enumerate(instance_ids)
        instance = df[df.num_instance .== inst, :]

        heuristics = groupby(instance, :num_heuristic)

        for heuristic in heuristics
            heuristic = sort(heuristic, :num_experiment)
            x = heuristic.num_experiment
            y = heuristic.last_sol
            plot!(figs[i], x, y, label = string(heuristic.heuristic_type[1]))  
            title!(figs[i], string("Last objective valie for instance ", i))
        end
    end

    for fig in figs
        display(fig)
    end
end

function plot_solving_time(csv_path::AbstractString)
    df = CSV.read(csv_path, DataFrame)


    instance_ids = unique(df.num_instance)

    figs = [plot() for _ in instance_ids]

    for (i,inst) in enumerate(instance_ids)
        instance = df[df.num_instance .== inst, :]

        heuristics = groupby(instance, :num_heuristic)

        for heuristic in heuristics
            heuristic = sort(heuristic, :num_experiment)
            x = heuristic.num_experiment
            y = heuristic.total_time
            plot!(figs[i], x, y, label = string(heuristic.heuristic_type[1]))  
            title!(figs[i], string("Total solving time for instance ", i))
        end
    end

    for fig in figs
        display(fig)
    end
end

function plot_node_visited(csv_path::AbstractString)
    df = CSV.read(csv_path, DataFrame)


    instance_ids = unique(df.num_instance)

    figs = [plot() for _ in instance_ids]

    for (i,inst) in enumerate(instance_ids)
        instance = df[df.num_instance .== inst, :]

        heuristics = groupby(instance, :num_heuristic)

        for heuristic in heuristics
            heuristic = sort(heuristic, :num_experiment)
            x = heuristic.num_experiment
            y = heuristic.node_visited
            plot!(figs[i], x, y, label = string(heuristic.heuristic_type[1]))  
            title!(figs[i], string("Node visited for instance ", i))
        end
    end

    for fig in figs
        display(fig)
    end
end