using BSON: @load, @save
using Dates
using Flux
using JSON
using SeaPearl
using Random
using ReinforcementLearning
const RL = ReinforcementLearning

include("agents.jl")
include("heuristics.jl")
include("tsptw_config.jl")
include("utils.jl")

experiment_config = TSPTWExperimentConfig(5, 10, 0, 100, 5, 200, 10, 1, 1, false)

tsptw_generator = SeaPearl.TsptwGenerator(
    experiment_config.num_cities,
    experiment_config.grid_size,
    experiment_config.max_tw_gap,
    experiment_config.max_tw,
    true
)

num_features = 20
featurizationType = SeaPearl.DefaultFeaturization

function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{featurizationType, TS}}) where TS
    return num_features
end


SR = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization, SeaPearl.DefaultTrajectoryState}
numInFeatures=SeaPearl.feature_length(SR)

SR = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization, SeaPearl.DefaultTrajectoryState}
num_input_features = SeaPearl.feature_length(SR)
reward_type = SeaPearl.GeneralReward
agent = build_tsptw_agent(num_input_features, experiment_config.num_cities)
values_raw = true
constraint_type = true
chosen_features = Dict([("values_raw", values_raw), ("constraint_type", constraint_type)])
learned_heuristic = SeaPearl.SimpleLearnedHeuristic{SR, reward_type, SeaPearl.FixedOutput}(agent; chosen_features=chosen_features)

featurizationType = SeaPearl.DefaultFeaturization
function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{featurizationType, TS}}) where TS
    return num_input_features
end

random_heuristics = []
for i in 1: experiment_config.num_random_heuristics
    push!(random_heuristics, SeaPearl.BasicHeuristic(select_random_value))
end
heuristic_min = SeaPearl.BasicHeuristic(select_min)
value_selection_array = [learned_heuristic, heuristic_min]
append!(value_selection_array, random_heuristics)
variable_selection = SeaPearl.MinDomainVariableSelection{false}()

function solve_tsptw_with_learning(
    experiment_config::TSPTWExperimentConfig, 
    value_selection_array::Array, 
    agent::RL.Agent, 
    learned_heuristic::SeaPearl.SimpleLearnedHeuristic,
    variable_selection
)

    metrics_array, eval_metrics_array=SeaPearl.train!(
        valueSelectionArray=value_selection_array,
        generator=tsptw_generator,
        nbEpisodes=experiment_config.num_episodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variable_selection,
        out_solver=true,
        verbose = false,
        evaluator=SeaPearl.SameInstancesEvaluator(
            value_selection_array, 
            tsptw_generator; 
            evalFreq=experiment_config.eval_freq, 
            nbInstances=experiment_config.num_instances
        ),
        restartPerInstances=experiment_config.num_restarts_per_instance
    )
    if experiment_config.save_artefacts    
        experiment_parameters = get_experiment_parameters(experiment_config, agent, learned_heuristic)
        save_experiment_artefacts(experiment_parameters, experiment_config, agent)
    end

    return metrics_array, eval_metrics_array
end

if abspath(PROGRAM_FILE) == @__FILE__
    metrics_array, eval_metrics_array = solve_tsptw_with_learning(experiment_config, value_selection_array, agent, learned_heuristic, variable_selection)
end