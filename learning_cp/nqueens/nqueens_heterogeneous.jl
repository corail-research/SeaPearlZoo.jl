using SeaPearl
using SeaPearlExtras
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux
using BSON: @save, @load
using JSON
using Random
using Dates
using Statistics

include("features.jl")

# -------------------
# Generator
# -------------------
board_size = 15
nqueens_generator = SeaPearl.NQueensGenerator(board_size)

featurizationType = SeaPearl.DefaultFeaturization
rewardType = SeaPearl.CPReward
SR_default = SeaPearl.DefaultStateRepresentation{featurizationType,SeaPearl.DefaultTrajectoryState}
SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{featurizationType,SeaPearl.HeterogeneousTrajectoryState}

# -------------------
# Internal variables
# -------------------
numInFeatures = 3
numInFeatures2 = [1, 5, 1]

# -------------------
# Experience variables
# -------------------
nbEpisodes = 10001
evalFreq = 500
nbInstances = 1

# -------------------
# Agent definition
# -------------------
include("agents_heterogeneous.jl")

# -------------------
# Value Heuristic definition
# -------------------

chosen_features = Dict(
    "constraint_activity" => false,
    "constraint_type" => true,
    "nb_involved_constraint_propagation" => false,
    "nb_not_bounded_variable" => false,
    "variable_domain_size" => false,
    "variable_initial_domain_size" => true,
    "variable_is_bound" => false,
    "values_onehot" => false,
    "values_raw" => true,
)

learnedHeuristic = SeaPearl.SimpleLearnedHeuristic{SR_default, rewardType, SeaPearl.FixedOutput}(agent)
learnedHeuristic2 = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous, rewardType, SeaPearl.FixedOutput}(agent2; chosen_features=chosen_features)

# Basic value-selection heuristic
selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)
nbRandomHeuristics = 0
valueSelectionArray = [learnedHeuristic, learnedHeuristic2, heuristic_min]

# -------------------
# Variable Heuristic definition
# -------------------
variableSelection = SeaPearl.MinDomainVariableSelection{false}()

# -------------------
# -------------------
# Core function
# -------------------
# -------------------

function trytrain(nbEpisodes::Int)
    experienceTime = now()
    dir = mkdir(string("exp_", Base.replace("$(round(experienceTime, Dates.Second(3)))", ":" => "-")))
    expParameters = Dict(
        :experimentParameters => Dict(
            :nbEpisodes => nbEpisodes,
            :evalFreq => evalFreq,
            :nbInstances => nbInstances,
        ),
        :generatorParameters => Dict(
            :boardSize => board_size,
        ),
        :nbRandomHeuristics => nbRandomHeuristics,
        :Featurization => Dict(
            :featurizationType => featurizationType,
            :chosen_features => chosen_features
        ),
        :learnerParameters => Dict(
            :model => string(agent.policy.learner.approximator.model),
            :gamma => agent.policy.learner.sampler.γ,
            :batch_size => agent.policy.learner.sampler.batch_size,
            :update_horizon => agent.policy.learner.sampler.n,
            :min_replay_history => agent.policy.learner.min_replay_history,
            :update_freq => agent.policy.learner.update_freq,
            :target_update_freq => agent.policy.learner.target_update_freq,
        ),
        :explorerParameters => Dict(
            :ϵ_stable => agent.policy.explorer.ϵ_stable,
            :decay_steps => agent.policy.explorer.decay_steps,
        ),
        :trajectoryParameters => Dict(
            :trajectoryType => typeof(agent.trajectory),
            :capacity => trajectory_capacity
        ),
        :reward => rewardType
    )
    open(dir * "/params.json", "w") do file
        JSON.print(file, expParameters)
    end
    cp("nqueens_heterogeneous.jl", dir*"/nqueens_heterogeneous.jl")
    cp("agents_heterogeneous.jl", dir*"/agents_heterogeneous.jl")
    metricsArray, eval_metricsArray = SeaPearl.train!(
        valueSelectionArray=valueSelectionArray,
        generator=nqueens_generator,
        nbEpisodes=nbEpisodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variableSelection,
        out_solver=true,
        verbose=true,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray, nqueens_generator; evalFreq=evalFreq, nbInstances=nbInstances),
        restartPerInstances=1
    )

    #saving model weights
    trained_weights = params(agent.policy.learner.approximator.model)
    @save dir * "/model_weights_nqueens_$(board_size).bson" trained_weights

    SeaPearlExtras.storedata(metricsArray[1]; filename=dir * "/nqueens_$(board_size)_training")
    SeaPearlExtras.storedata(metricsArray[2]; filename=dir * "/nqueens_$(board_size)_training2")
    SeaPearlExtras.storedata(eval_metricsArray[:, 1]; filename=dir * "/nqueens_$(board_size)_trained")
    SeaPearlExtras.storedata(eval_metricsArray[:, 2]; filename=dir * "/nqueens_$(board_size)_trained2")
    SeaPearlExtras.storedata(eval_metricsArray[:, 3]; filename=dir * "/nqueens_$(board_size)_min")

    return metricsArray, eval_metricsArray
end



# -------------------
# -------------------

metricsArray, eval_metricsArray = trytrain(nbEpisodes)
nothing
