using SeaPearl
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using BSON: @save, @load
using Random
using Statistics
using Dates
using JSON

include("../experiment_setup.jl")
include("../features.jl")
include("../model_config.jl")
include("../models.jl")
include("../rewards.jl")

include("knapsack_models_ppo.jl")
include("argparse_knapsack_ppo.jl")


knapsack_generator, experiment_setup, knapsack_agent_config, csv_path = set_settings()

StateRepresentation = SeaPearl.DefaultStateRepresentation{KnapsackFeaturization, SeaPearl.DefaultTrajectoryState}
numInFeatures = SeaPearl.feature_length(StateRepresentation)

numInFeatures = 16
actor_model = build_knapsack_actor_model(knapsack_agent_config.output_size)
critic_model = build_knapsack_critic_model(knapsack_agent_config.output_size)
agent = build_knapsack_ppo_agent(actor_model, critic_model, knapsack_agent_config)

# Value Heuristic definition
learnedHeuristic = SeaPearl.SimpleLearnedHeuristic{StateRepresentation, knapsackReward, SeaPearl.FixedOutput}(agent)
basicHeuristic = SeaPearl.BasicHeuristic((x; cpmodel=nothing) -> SeaPearl.maximum(x.domain)) 

# Variable Heuristic definition
struct KnapsackVariableSelection <: SeaPearl.AbstractVariableSelection{false} end

function (::KnapsackVariableSelection)(model::SeaPearl.CPModel)
    i = 1
    while SeaPearl.isbound(model.variables["x[" * string(i) * "]"])
        i += 1
    end
    return model.variables["x[" * string(i) * "]"]
end

valueSelectionArray = [learnedHeuristic, basicHeuristic]

function solve_knapsack_with_learning!(experiment_setup::KnapsackExperimentConfig, save_experiment_artefacts::Bool=false)
    experiment_parameters = Dict(
        :nbEpisodes => experiment_setup.num_episodes,
        :evalFreq => experiment_setup.eval_freq,
        :nbInstances => experiment_setup.num_instances
    )
    metricsArray, eval_metricsArray = SeaPearl.train!(;
        valueSelectionArray= valueSelectionArray,
        generator=knapsack_generator,
        nbEpisodes=experiment_setup.num_episodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=KnapsackVariableSelection(),
        out_solver=false,
        verbose=false,
        evaluator=SeaPearl.SameInstancesEvaluator(
            valueSelectionArray, 
            knapsack_generator; 
            evalFreq=experiment_setup.eval_freq, 
            nbInstances=experiment_setup.num_instances
        ),
        restartPerInstances = 1
    )
    if save_experiment_artefacts
        experience_time = now()
        dir = mkdir(string("exp_",Base.replace("$(round(experience_time, Dates.Second(3)))",":"=>"-")))
        open(dir*"/params.json", "w") do file
            JSON.print(file, experiment_parameters)
        end    
        trained_weights = params(agent.policy.learner.approximator.model)
        @save dir*"/model_weights_knapsack.bson" trained_weights
    end

    return metricsArray, eval_metricsArray
end

# if abspath(PROGRAM_FILE) == @__FILE__
metricsArray, eval_metricsArray = solve_knapsack_with_learning!(experiment_setup)
# end

include("../../utils/save_metrics.jl")

if save_performance
    save_metrics(eval_metricsArray, csv_path, eval_freq=experiment_setup.eval_freq)
end