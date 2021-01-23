using SeaPearl
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux

using Plots
gr()

include("rewards.jl")
include("features.jl")

# -------------------
# Generator
# -------------------
coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(30, 5, 0.5)

# -------------------
# Internal variables
# -------------------
numInFeatures = 6 + coloring_generator.n
maxNumberOfCPNodes = 1 + floor(Int64, coloring_generator.n * ( 3 + coloring_generator.p ))*5
state_size = (maxNumberOfCPNodes, numInFeatures + maxNumberOfCPNodes + 2 + 1)

# -------------------
# Experience variables
# -------------------
nb_episodes = 600
eval_freq = 30

# -------------------
# Agent definition
# -------------------
include("agents.jl")
learnedHeuristic = SeaPearl.LearnedHeuristic{SeaPearl.DefaultStateRepresentation{BetterFeaturization}, InspectReward, SeaPearl.FixedOutput}(agent, maxNumberOfCPNodes)

# Basic value-selection heuristic
selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)

# -------------------
# Variable selection
# -------------------
variableSelection = SeaPearl.MinDomainVariableSelection{false}()



# -------------------
# -------------------
# TRAINING
# -------------------
# -------------------
bestsolutions, nodevisited, timeneeded, eval_nodes, eval_tim = SeaPearl.train!(
    valueSelectionArray=[learnedHeuristic, heuristic_min], 
    generator=coloring_generator,
    nb_episodes=nb_episodes,
    strategy=SeaPearl.DFSearch,
    variableHeuristic=variableSelection,
    out_solver=false,
    verbose = false,
    evaluator=SeaPearl.SameInstancesEvaluator(; eval_freq = eval_freq, nb_instances = 10)
)


