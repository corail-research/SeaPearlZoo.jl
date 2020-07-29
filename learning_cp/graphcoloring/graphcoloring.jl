using SeaPearl
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux

using Plots
gr()

####
include("rewards.jl")
include("features.jl")
####

coloring_generator = SeaPearl.LegacyGraphColoringGenerator(10, 1.4)

numInFeatures = 16
numberOfCPNodes = 1 + floor(Int64, coloring_generator.nb_nodes * ( 3 + coloring_generator.density ))
#numberOfCPNodes = 141

state_size = (numberOfCPNodes, numInFeatures + numberOfCPNodes + 2 + 1)

include("agents.jl")
learnedHeuristic = SeaPearl.LearnedHeuristic{SeaPearl.DefaultStateRepresentation{BetterFeaturization}, InspectReward, SeaPearl.FixedOutput}(agent)

selectMin(x::SeaPearl.IntVar) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)

variableSelection = SeaPearl.RandomVariableSelection()

############# TRAIN

bestsolutions, nodevisited, timeneeded = SeaPearl.train!(
    valueSelectionArray=[learnedHeuristic, heuristic_min], 
    generator=coloring_generator,
    nb_episodes=400,
    strategy=SeaPearl.DFSearch,
    variableHeuristic=variableSelection,
    verbose = false
)

# plot 
a, b = size(nodevisited)
x = 1:a

p1 = plot(x, nodevisited, xlabel="Episode", ylabel="Number of nodes visited", ylims = [0, 200])

############# BENCHMARK

#= bestsolutions, nodevisited, timeneeded = SeaPearl.benchmark_solving(
    valueSelectionArray=[learnedHeuristic, heuristic_min], 
    generator=coloring_generator,
    nb_episodes=200,
    strategy=SeaPearl.DFSearch,
    variableHeuristic=variableSelection,
    verbose = false
) =#

# plot 
a, b = size(nodevisited)
x = 1:a

p2 = plot(x, nodevisited, xlabel="Episode", ylabel="Number of nodes visited", ylims = [0, 200])
p3 = plot(x, timeneeded, xlabel="Episode", ylabel="Time needed", ylims = [0, 0.01])


p = plot(p1, p2, p3, legend = false, layout = (3, 1))

display(p)
