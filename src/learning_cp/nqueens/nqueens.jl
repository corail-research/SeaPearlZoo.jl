using BSON: @save, @load
using Dates
using Flux
using JSON
using LightGraphs
using Random
using ReinforcementLearning
const RL = ReinforcementLearning
using SeaPearl
using Statistics

include("agents.jl")
include("model_config.jl")
include("nqueens_config.jl")
include("rewards.jl")
include("utils.jl")

board_size = 15

struct BetterFeaturization <: SeaPearl.AbstractFeaturization end

function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{BetterFeaturization})
    g = sr.cplayergraph
    features = zeros(Float32, 2 * board_size + 1, LightGraphs.nv(g))
    for i in 1:LightGraphs.nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            if isa(cp_vertex.variable, SeaPearl.IntVarViewOffset)
                id = cp_vertex.variable.id
                if occursin("+",id)
                    features[parse(Int, split(id,"+")[end]), i] = 1
                else
                    features[parse(Int, split(id,"-")[end]), i] = -1
                end
            else
                id = cp_vertex.variable.id
                features[string_to_queen(id), i] = 1
            end
        end
        if isa(cp_vertex, SeaPearl.ConstraintVertex)
            features[board_size + 1, i] = 1
        end
        if isa(cp_vertex, SeaPearl.ValueVertex)
            features[3, i] = 1.
            value = cp_vertex.value
            features[board_size + 1 + value, i] = 1.
        end
    end
    features
end

function string_to_queen(id::String)::Int
    parse(Int, split(id,"_")[end])
end

function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{BetterFeaturization, TS}}) where TS
    return 1 + 2 * board_size
end

features_type = BetterFeaturization
SR = SeaPearl.DefaultStateRepresentation{features_type, SeaPearl.DefaultTrajectoryState}
experiment_config = NQueensConfig(
    board_size, 
    SeaPearl.feature_length(SR), 
    10000,
    1000,
    50,
    0,
    1,
    SeaPearl.CPReward,
    false
)
model_config = ModelConfig(SeaPearl.feature_length(SR), experiment_config.board_size, false)
approximator_model = build_model(model_config)
target_approximator_model = build_model(model_config)
agent_config = AgentConfig(
    approximator_model,
    target_approximator_model,
    32,
    12,
    256,
    1,
    200,
    1.0,
    0.1,
    :exp,
    50000,
    1,
    50000,
    board_size
)
agent = build_agent(agent_config)

learned_heuristic_config = LearnedHeuristicConfig(1., 0.1 , 50, 50)
learned_heuristic = SeaPearl.SupervisedLearnedHeuristic{SR, experiment_config.reward_type, SeaPearl.FixedOutput}(
    agent, 
    eta_init=learned_heuristic_config.eta_init,
    eta_stable=learned_heuristic_config.eta_stable, 
    warmup_steps=learned_heuristic_config.warmup_steps, 
    decay_steps=learned_heuristic_config.decay_steps,
    rng=MersenneTwister(1234)
)

selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)
random_heuristics = []
for i in 1 : experiment_config.num_random_heuristics
    push!(random_heuristics, SeaPearl.BasicHeuristic(select_random_value))
end

value_selection_array = [learned_heuristic, heuristic_min]
append!(value_selection_array, random_heuristics)
variable_selection = SeaPearl.MinDomainVariableSelection{false}()

function solve_learning_nqueens(
    experiment_config::NQueensConfig, 
    agent::RL.Agent, 
    learned_heuristic::SeaPearl.SupervisedLearnedHeuristic,
    variable_selection::SeaPearl.MinDomainVariableSelection,
    value_selection_array
)
    if experiment_config.save_experiment_artefacts
        experiment_parameters = build_experiment_parameters_dict(experiment_config, learned_heuristic, agent)
        experiment_time = now()
        dir = mkdir(string("exp_", Base.replace("$(round(experiment_time, Dates.Second(3)))", ":" => "-")))
        open(dir * "/params.json", "w") do file
            JSON.print(file, experiment_parameters)
        end
    end
    instance_evaluator = SeaPearl.SameInstancesEvaluator(
        value_selection_array, 
        experiment_config.generator; 
        evalFreq=experiment_config.eval_freq, 
        nbInstances=experiment_config.num_instances
    )
    metrics_array, eval_metrics_array = SeaPearl.train!(
        valueSelectionArray=value_selection_array,
        generator=experiment_config.generator,
        nbEpisodes=experiment_config.num_episodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variable_selection,
        out_solver=true,
        verbose=false,
        evaluator=instance_evaluator,
        restartPerInstances=experiment_config.num_restarts_per_instance
    )
    if experiment_config.save_experiment_artefacts
        trained_weights = params(agent.policy.learner.approximator.model)
        @save dir * "/model_weights_nqueens_$(experimen_config.board_size).bson" trained_weights
    end

    return metrics_array, eval_metrics_array
end

if abspath(PROGRAM_FILE) == @__FILE__
    metrics_array, eval_metrics_array = solve_learning_nqueens(experiment_config, agent, learned_heuristic, variable_selection, value_selection_array)
end