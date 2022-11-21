using BSON: @save, @load
using Random
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux
using SeaPearl
using SeaPearlExtras

include("rewards.jl")
include("features.jl")


eternity2_generator = SeaPearl.Eternity2Generator(6,6,6)

# -------------------
# Internal variables
# -------------------
SR = SeaPearl.DefaultStateRepresentation{EternityFeaturization, SeaPearl.DefaultTrajectoryState}
num_input_features = SeaPearl.feature_length(SR)
num_global_features = SeaPearl.global_feature_length(SR)
# -------------------
# Experience variables
# -------------------
num_episodes = 100
eval_freq = 30
num_instances = 1
num_random_heuristics = 1

# -------------------
# Agent definition
# -------------------

include("agents.jl")


learned_heuristic = SeaPearl.LearnedHeuristic{SR, SeaPearl.CPReward, SeaPearl.FixedOutput}(agent)
selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
min_domain_heuristic = SeaPearl.BasicHeuristic(selectMin)

function select_random_value(x::SeaPearl.IntVar; cpmodel=nothing)
    selected_number = rand(1:length(x.domain))
    i = 1
    for value in x.domain
        if i == selected_number
            return value
        end
        i += 1
    end
    @assert false "This should not happen"
end

random_heuristics = []
for i in 1:num_random_heuristics
    push!(random_heuristics, SeaPearl.BasicHeuristic(select_random_value))
end

value_selection_array = [learned_heuristic, min_domain_heuristic]
append!(value_selection_array, random_heuristics)
variable_selection = SeaPearl.MinDomainVariableSelection{false}()


function trytrain(num_episodes::Int)

    instance_evaluator = SeaPearl.SameInstancesEvaluator(
        value_selection_array,
        eternity2_generator; 
        evalFreq=eval_freq, 
        nbInstances=num_instances
    )
    metrics_array, eval_metrics_array = SeaPearl.train!(
        valueSelectionArray=value_selection_array,
        generator=eternity2_generator,
        nbEpisodes=num_episodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variable_selection,
        out_solver=false,
        verbose = true,
        evaluator=instance_evaluator,
        restartPerInstances = 1
    )

    return metrics_array, eval_metrics_array
end



# -------------------
# -------------------

metrics_array, eval_metrics_array = trytrain(num_episodes)