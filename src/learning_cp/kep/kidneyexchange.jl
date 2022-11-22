using Flux
# using GeometricFlux
using LightGraphs
using Random
using ReinforcementLearning
const RL = ReinforcementLearning
using SeaPearl
using Statistics

include("agents.jl")
include("kep_config.jl")
include("features.jl")
include("train_kep_agent.jl")
include("utils.jl")

function main_learning_kep(args::KepParameters)
    kep_generator = SeaPearl.KepGenerator(args.num_nodes, args.density)
    kep_eval_generator = SeaPearl.KepGenerator(args.num_nodes, args.density)
    SR = SeaPearl.DefaultStateRepresentation{KepFeaturization, SeaPearl.DefaultTrajectoryState}
    args.SR = SR
    args.num_input_features = SeaPearl.feature_length(SR)
    agent = create_agent(args)
    learned_heuristic = SeaPearl.SimpleLearnedHeuristic{SR, args.reward, SeaPearl.FixedOutput}(agent)
    basic_heuristic = SeaPearl.BasicHeuristic()
    random_generator = get_random_generator_from_seed(args.seed)
    random_heuristics = []
    
    for i in 1:args.num_random_heuristics
        push!(random_heuristics, SeaPearl.BasicHeuristic(select_random_value_kep(random_generator)))
    end

    value_selection_array = [learned_heuristic, basic_heuristic]
    append!(value_selection_array, random_heuristics)
    variable_selection = SeaPearl.MinDomainVariableSelection()
    train_kep_agent(args, kep_generator, kep_eval_generator, value_selection_array, variable_selection, agent, false)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_learning_kep(KepParameters())
end