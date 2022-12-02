using SeaPearl

include("utils.jl")

function train_kep_agent(
        args::KepParameters,
        generator::SeaPearl.KepGenerator,      
        eval_generator::SeaPearl.KepGenerator,    
        heuristics::Vector{<:SeaPearl.ValueSelection},
        variable_selection::SeaPearl.AbstractVariableSelection,
        agent::RL.Agent,
        save_experiment_artefacts::Bool
    )

    if save_experiment_artefacts
        directory = save_experiment_config(args)
    end

    metrics_array, eval_metrics_array = SeaPearl.train!(
        valueSelectionArray = heuristics,
        generator = generator,
        nbEpisodes = args.num_episodes,
        strategy = args.strategy,
        eval_strategy = args.eval_strategy,
        variableHeuristic = variable_selection,
        out_solver = true,
        verbose = true,
        evaluator = SeaPearl.SameInstancesEvaluator(heuristics, eval_generator;
            evalFreq = div(args.num_episodes, args.num_evals), 
            nbInstances = args.num_instances, 
            evalTimeOut = args.eval_timeout
        ),
        metrics = nothing, 
        restartPerInstances = args.num_restarts_per_instance,
    )
    if save_experiment_artefacts
        save_experiment_weights(agent, directory)
    end

    return metrics_array, eval_metrics_array
end