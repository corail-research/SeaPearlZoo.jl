using SeaPearl
# using SeaPearlExtras
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
# using GeometricFlux
using BSON: @save, @load
using Random
using Statistics
using Dates
using JSON

include("agents.jl")
include("features.jl")
include("rewards.jl")

knapsack_generator = SeaPearl.KnapsackGenerator(10, 10, 0.2)
StateRepresentation = SeaPearl.DefaultStateRepresentation{KnapsackFeaturization, SeaPearl.DefaultTrajectoryState}
numInFeatures = SeaPearl.feature_length(StateRepresentation)
experiment_setup = KnapsackExperimentConfig(1000, 100, 3, 0)

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
        :nbEpisodes => experiment_setup.nbEpisodes,
        :evalFreq => experiment_setup.evalFreq,
        :nbInstances => experiment_setup.nbInstances
    )
    metricsArray, eval_metricsArray = SeaPearl.train!(;
        valueSelectionArray= valueSelectionArray,
        generator=knapsack_generator,
        nbEpisodes=experiment_setup.nbEpisodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=KnapsackVariableSelection(),
        out_solver=false,
        verbose=false,
        evaluator=SeaPearl.SameInstancesEvaluator(
            valueSelectionArray, 
            knapsack_generator; 
            evalFreq=experiment_setup.evalFreq, 
            nbInstances=experiment_setup.nbInstances
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

    # SeaPearlExtras.storedata(metricsArray[1]; filename=dir*"/knapsack_training")
    # SeaPearlExtras.storedata(eval_metricsArray[:,1]; filename=dir*"/knapsack_learned")
    # SeaPearlExtras.storedata(eval_metricsArray[:,2]; filename=dir*"/knapsack_basic")
    return metricsArray, eval_metricsArray
end

metricsArray, eval_metricsArray = solve_knapsack_with_learning!(experiment_setup)
nothing
