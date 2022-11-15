using SeaPearl
import Pkg
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
# using GeometricFlux
# using JSON
# using BSON: @save, @load
using Dates
using Random
using LightGraphs

include("agent_config.jl")
include("coloring_config.jl")
# include("features.jl")
include("models.jl")

# -------------------
# Internal variables
# -------------------
# featurizationType = BetterFeaturization
coloring_settings = ColoringSettings(1, 1, 100, 50, 1, 20, 5, 0.95)
rewardType = SeaPearl.CPReward
SR = SeaPearl.DefaultStateRepresentation{SeaPearl.AbstractFeaturization, SeaPearl.DefaultTrajectoryState}
numInFeatures = SeaPearl.feature_length(SR)
instance_generator = SeaPearl.BarabasiAlbertGraphGenerator(coloring_settings.nbNodes, coloring_settings.nbMinColor)

function SeaPearl.featurize(
    sr::SeaPearl.DefaultStateRepresentation{SeaPearl.AbstractFeaturization,TS}
    ) where TS

    g = sr.cplayergraph
    features = zeros(Float32, instance_generator.n + 6, nv(g))
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            features[1,i] = 1.
            if g.cpmodel.objective == cp_vertex.variable
                features[6, i] = 1.
            end
        end
        if isa(cp_vertex, SeaPearl.ConstraintVertex)
            features[2, i] = 1.
            constraint = cp_vertex.constraint
            if isa(constraint, SeaPearl.NotEqual)
                features[4, i] = 1.
            end
            if isa(constraint, SeaPearl.LessOrEqual)
                features[5, i] = 1.
            end
        end
        if isa(cp_vertex, SeaPearl.ValueVertex)
            features[3, i] = 1.
            value = cp_vertex.value
            features[6+value, i] = 1.
        end
    end
    features
end

function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{SeaPearl.AbstractFeaturization, TS}}) where TS
    instance_generator.n + 6
end

function SeaPearl.global_feature_length(::Type{SeaPearl.DefaultStateRepresentation{SeaPearl.AbstractFeaturization, TS}}) where TS
    return 0
end
# -------------------
# Value Heuristic definition
# -------------------
agent_config = AgentConfig(0.99f0, 16, 4 128, 1, 200, 3000)
approximator_model = build_graph_coloring_approximator_model(instance_generator.n)
target_approximator_model = build_graph_coloring_target_approximator_model(instance_generator.n)
agent = build_graph_coloring_agent(approximator_model, target_approximator_model, agent_config)
learnedHeuristic = SeaPearl.SimpleLearnedHeuristic{SR, rewardType, SeaPearl.FixedOutput}(agent)

# Basic value-selection heuristic
selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)
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

randomHeuristics = []
for i in 1:coloring_settings.nbRandomHeuristics
    push!(randomHeuristics, SeaPearl.BasicHeuristic(select_random_value))
end

valueSelectionArray = [learnedHeuristic, heuristic_min]
append!(valueSelectionArray, randomHeuristics)
variableSelection = SeaPearl.MinDomainVariableSelection{false}() # Variable Heuristic definition

# -------------------
# Core function
# -------------------

function solve_learning_coloring(coloring_settings::ColoringSettings, instance_generator::SeaPearl.AbstractModelGenerator)
    experienceTime = now()
    dir = mkdir(string("exp_",Base.replace("$(round(experienceTime, Dates.Second(3)))",":"=>"-")))
    expParameters = Dict(
        :experimentParameters => Dict(
            :nbEpisodes => coloring_settings.nbEpisodes ,
            :restartPerInstances => coloring_settings.restartPerInstances,
            :evalFreq => coloring_settings.evalFreq,
            :nbInstances => coloring_settings.nbInstances,
        ),
        :generatorParameters => Dict(
            :nbNodes => coloring_settings.nbNodes,
            :nbMinColor => coloring_settings.nbMinColor,
            :density => coloring_settings.density
        ),
        :nbRandomHeuristics => nbRandomHeuristics,
        :Featurization => Dict(
            :featurizationType => featurizationType,
            #:chosen_features => featurizationType == SeaPearl.FeaturizationHelper ? chosen_features : nothing
            :chosen_features => nothing
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
    # open(dir*"/params.json", "w") do file
    #     JSON.print(file, expParameters)
    # end

    metricsArray, eval_metricsArray = SeaPearl.train!(
        valueSelectionArray=valueSelectionArray,
        generator=instance_generator,
        nbEpisodes=coloring_settings.nbEpisodes ,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variableSelection,
        out_solver=true,
        verbose = false,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray,instance_generator; evalFreq = coloring_settings.evalFreq, nbInstances = coloring_settings.nbInstances),
        restartPerInstances = coloring_settings.restartPerInstances
    )

    #saving model weights
    model = agent.policy.learner.approximator
    @save dir*"/model_gc"*string(instance_generator.n)*".bson" model

    # SeaPearlExtras.storedata(metricsArray[1]; filename=dir*"/graph_coloring_$(coloring_settings.nbNodes)_traininglearned")
    # SeaPearlExtras.storedata(metricsArray[2]; filename=dir*"/graph_coloring_$(coloring_settings.nbNodes)_traininggreedy")
    # for i = 1:nbRandomHeuristics
        # SeaPearlExtras.storedata(metricsArray[2+i]; filename=dir*"/graph_coloring_$(coloring_settings.nbNodes)_trainingrandom$(i)")
    # end
    # SeaPearlExtras.storedata(eval_metricsArray[:,1]; filename=dir*"/graph_coloring_$(coloring_settings.nbNodes)_learned")
    # SeaPearlExtras.storedata(eval_metricsArray[:,2]; filename=dir*"/graph_coloring_$(coloring_settings.nbNodes)_greedy")
    # for i = 1:nbRandomHeuristics
    #     SeaPearlExtras.storedata(eval_metricsArray[:,i+2]; filename=dir*"/graph_coloring_$(coloring_settings.nbNodes)_random$(i)")
    # end

    return metricsArray, eval_metricsArray
end

metricsArray, eval_metricsArray = trytrain(coloring_settings, instance_generator)
nothing