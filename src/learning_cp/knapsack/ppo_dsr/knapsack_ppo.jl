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

nb_items = 10
max_weight = 10
correlation = 0.2
knapsack_generator = SeaPearl.KnapsackGenerator(nb_items, max_weight, correlation)

StateRepresentation = SeaPearl.DefaultStateRepresentation{KnapsackFeaturization, SeaPearl.DefaultTrajectoryState}
numInFeatures = SeaPearl.feature_length(StateRepresentation)

num_episodes = 100
eval_freq = 10
num_instances = 20
num_random_heuristics = 0
experiment_setup = KnapsackExperimentConfig(num_episodes, eval_freq, num_instances, num_random_heuristics)

gamma = 0.99f0
lambda = 0.95f0
clip_range = 0.2f0
max_grad_norm = 0.5f0
n_epochs = 10
n_microbatches = 32
actor_loss_weight = 1.0f0
critic_loss_weight = 0.5f0
entropy_loss_weight = 0.00f0
output_size = 2
update_freq = 128 #2048
trajectory_capacity = update_freq

knapsack_agent_config = KnapsackPPOAgentConfig(gamma, lambda, clip_range, max_grad_norm, n_epochs, n_microbatches, actor_loss_weight, critic_loss_weight, entropy_loss_weight, output_size, update_freq, trajectory_capacity)
actor_model = build_knapsack_actor_model(output_size)
critic_model = build_knapsack_critic_model(output_size)
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