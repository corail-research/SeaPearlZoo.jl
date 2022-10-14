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

# -------------------
# Generator
# -------------------

n_city = 10
grid_size = 25
max_tw_gap = 0
max_tw = 100

featurizationType = SeaPearl.DefaultFeaturization
rewardType = SeaPearl.GeneralReward
tsptw_generator = SeaPearl.TsptwGenerator(n_city, grid_size, max_tw_gap, max_tw, true)
SR_default = SeaPearl.DefaultStateRepresentation{featurizationType,SeaPearl.DefaultTrajectoryState}
SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{featurizationType,SeaPearl.HeterogeneousTrajectoryState}

# -------------------
# Internal variables
# -------------------
numInFeatures = 3
numInFeatures2 = [1, 16, 1]

# -------------------
# Experience variables
# -------------------
nbEpisodes = 2001
evalFreq = 200
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
include("nearest_heuristic.jl")
nearest_heuristic = SeaPearl.BasicHeuristic(select_nearest_neighbor) # Basic value-selection heuristic
nbRandomHeuristics = 0
valueSelectionArray = [learnedHeuristic, learnedHeuristic2, nearest_heuristic]

# -------------------
# Variable Heuristic definition
# -------------------
struct TsptwVariableSelection{TakeObjective} <: SeaPearl.AbstractVariableSelection{TakeObjective} end
TsptwVariableSelection(;take_objective=false) = TsptwVariableSelection{take_objective}()
function (::TsptwVariableSelection{false})(cpmodel::SeaPearl.CPModel; rng=nothing)
    for i in 1:length(keys(cpmodel.variables))
        if haskey(cpmodel.variables, "a_"*string(i)) && !SeaPearl.isbound(cpmodel.variables["a_"*string(i)])
            return cpmodel.variables["a_"*string(i)]
        end
    end
end
variableSelection = TsptwVariableSelection()

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
            :nCity => n_city,
            :gridSize => grid_size,
            :maxTwGap => max_tw_gap,
            :maxTw => max_tw,
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
    cp("graphcoloring_heterogeneous.jl", dir*"/graphcoloring_heterogeneous.jl")
    cp("agents_heterogeneous.jl", dir*"/agents_heterogeneous.jl")
    metricsArray, eval_metricsArray = SeaPearl.train!(
        valueSelectionArray=valueSelectionArray,
        generator=tsptw_generator,
        nbEpisodes=nbEpisodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variableSelection,
        out_solver=true,
        verbose=true,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray, tsptw_generator; evalFreq=evalFreq, nbInstances=nbInstances),
        restartPerInstances=1
    )

    #saving model weights
    trained_weights = params(agent.policy.learner.approximator.model)
    @save dir * "/model_weights_tsptw_$(n_city).bson" trained_weights

    SeaPearlExtras.storedata(metricsArray[1]; filename=dir * "/tsptw_$(n_city)_training")
    SeaPearlExtras.storedata(metricsArray[2]; filename=dir * "/tsptw_$(n_city)_training2")
    SeaPearlExtras.storedata(eval_metricsArray[:, 1]; filename=dir * "/tsptw_$(n_city)_trained")
    SeaPearlExtras.storedata(eval_metricsArray[:, 2]; filename=dir * "/tsptw_$(n_city)_trained2")
    SeaPearlExtras.storedata(eval_metricsArray[:, 3]; filename=dir * "/tsptw_$(n_city)_min")

    return metricsArray, eval_metricsArray
end



# -------------------
# -------------------

metricsArray, eval_metricsArray = trytrain(nbEpisodes)
nothing
