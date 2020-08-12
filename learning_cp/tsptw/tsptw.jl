using SeaPearl
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux

using Plots
gr()

####
include("rewards.jl")
####

n_city = 41
grid_size = 100
max_tw_gap = 10
max_tw = 100

numInFeatures = 6

tsptw_generator = SeaPearl.TsptwGenerator(n_city, grid_size, max_tw_gap, max_tw)

# numberOfCPNodes = 1 + floor(Int64, coloring_generator.nb_nodes * ( 3 + coloring_generator.density ))
#numberOfCPNodes = 141

state_size = (n_city, n_city+6+2)

include("agents.jl")
learnedHeuristic = SeaPearl.LearnedHeuristic{SeaPearl.TsptwStateRepresentation, SeaPearl.TsptwReward, SeaPearl.VariableOutput}(agent)

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

eval_freq = 1000

bestsolutions, nodevisited, timeneeded, eval_nodes, eval_tim = SeaPearl.train!(
    valueSelectionArray=[learnedHeuristic, heuristic_min], 
    generator=tsptw_generator,
    nb_episodes=3000,
    strategy=SeaPearl.DFSearch,
    variableHeuristic=variableSelection,
    verbose = false,
    out_solver=true,
    evaluator=SeaPearl.SameInstancesEvaluator(; eval_freq = eval_freq, nb_instances = 5)
)

# plot 
a, b = size(nodevisited)
x = 1:a

p1 = plot(x, nodevisited, xlabel="Episode", ylabel="Number of nodes visited", ylims = [0, 1000])

p5 = plot(1:(floor(Int64, a/eval_freq)), eval_nodes[1:floor(Int64, a/eval_freq), :], xlabel="Evaluation", ylabel="Number of nodes visited", ylims = [0, 20000])
display(p5)

############# BENCHMARK

#= bestsolutions, nodevisited, timeneeded = SeaPearl.benchmark_solving(
    valueSelectionArray=[learnedHeuristic, heuristic_min], 
    generator=coloring_generator,
    nb_episodes=200,
    strategy=SeaPearl.DFSearch,
    variableHeuristic=variableSelection,
    verbose = false
# ) =#

# # plot 
# a, b = size(nodevisited)
# x = 1:a

# p2 = plot(x, nodevisited, xlabel="Episode", ylabel="Number of nodes visited", ylims = [0, 200])
# p3 = plot(x, timeneeded, xlabel="Episode", ylabel="Time needed", ylims = [0, 0.01])


# p = plot(p1, p2, p3, legend = false, layout = (3, 1))

# display(p)
