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

n_city = 10
grid_size = 40
max_tw_gap = 10
max_tw = 20

tsptw_generator = SeaPearl.TsptwGenerator(n_city, grid_size, max_tw_gap, max_tw)

numInFeatures = 27
# numberOfCPNodes = 1 + floor(Int64, coloring_generator.nb_nodes * ( 3 + coloring_generator.density ))
#numberOfCPNodes = 141
maxNumberOfCPNodes = 1000

state_size = (maxNumberOfCPNodes, numInFeatures + maxNumberOfCPNodes + 2 + 1)

include("agents.jl")
learnedHeuristic = SeaPearl.LearnedHeuristic{SeaPearl.DefaultStateRepresentation{BetterFeaturization}, InspectReward, SeaPearl.FixedOutput}(agent, maxNumberOfCPNodes)

selectMin(x::SeaPearl.IntVar) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)

struct TsptwVariableSelection{TakeObjective} <: SeaPearl.AbstractVariableSelection{TakeObjective} end

TsptwVariableSelection(;take_objective=false) = TsptwVariableSelection{take_objective}()

function (::TsptwVariableSelection{false})(cpmodel::SeaPearl.CPModel; rng=nothing)
    for i in 1:length(keys(cpmodel.variables))
        if haskey(cpmodel.variables, "a_"*string(i)) && !SeaPearl.isbound(cpmodel.variables["a_"*string(i)])
            return cpmodel.variables["a_"*string(i)]
        end
    end
    println(cpmodel.variables)
end

variableSelection = TsptwVariableSelection()

############# TRAIN

bestsolutions, nodevisited, timeneeded = SeaPearl.train!(
    valueSelectionArray=[learnedHeuristic, heuristic_min], 
    generator=tsptw_generator,
    nb_episodes=200,
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
