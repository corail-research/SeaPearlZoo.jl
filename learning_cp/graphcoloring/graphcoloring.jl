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

coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(30, 5, 0.5)

numInFeatures = 6 + coloring_generator.n
maxNumberOfCPNodes = 1 + floor(Int64, coloring_generator.n * ( 3 + coloring_generator.p ))*5
#numberOfCPNodes = 141

nb_episodes = 600
eval_freq = 30


state_size = (maxNumberOfCPNodes, numInFeatures + maxNumberOfCPNodes + 2 + 1)
println("state_size", state_size)

include("agents.jl")
learnedHeuristic = SeaPearl.LearnedHeuristic{SeaPearl.DefaultStateRepresentation{BetterFeaturization}, InspectReward, SeaPearl.FixedOutput}(agent, maxNumberOfCPNodes)

selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)

variableSelection = SeaPearl.MinDomainVariableSelection{false}()

maxNumberOfSteps = nb_episodes
total_rewards = zeros(maxNumberOfSteps)
numberOfSteps = 1
function metricsFun(;kwargs...)
    if numberOfSteps < maxNumberOfSteps && isa(kwargs[:heuristic], SeaPearl.LearnedHeuristic)
        total_rewards[numberOfSteps] = kwargs[:total_reward]
        global numberOfSteps += 1
    end
end

############# TRAIN

bestsolutions, nodevisited, timeneeded, eval_nodes, eval_tim = SeaPearl.train!(
    valueSelectionArray=[learnedHeuristic, heuristic_min], 
    generator=coloring_generator,
    nb_episodes=nb_episodes,
    strategy=SeaPearl.DFSearch,
    variableHeuristic=variableSelection,
    out_solver=false,
    verbose = false,
    metricsFun = metricsFun,
    evaluator=SeaPearl.SameInstancesEvaluator(; eval_freq = eval_freq, nb_instances = 10)
)

# plot 
a, b = size(nodevisited)
x = 1:a

p1 = plot(x, nodevisited, xlabel="Episode", ylabel="Number of nodes visited", ylims = [0, 1000])
display(p1)
p5 = plot(1:(floor(Int64, a/eval_freq)), eval_nodes[1:floor(Int64, a/eval_freq), :], xlabel="Evaluation", ylabel="Number of nodes visited", ylims = [0, 200])
display(p5)
############# BENCHMARK

# bestsolutions, nodevisited, timeneeded = SeaPearl.benchmark_solving(
#     valueSelectionArray=[learnedHeuristic, heuristic_min], 
#     generator=coloring_generator,
#     nb_episodes=200,
#     strategy=SeaPearl.DFSearch,
#     variableHeuristic=variableSelection,
#     verbose = false
# )

# # plot 
# a, b = size(nodevisited)
# x = 1:a

# p2 = plot(x, nodevisited, xlabel="Episode", ylabel="Number of nodes visited", ylims = [0, 200])
# p3 = plot(x, timeneeded, xlabel="Episode", ylabel="Time needed", ylims = [0, 0.01])


# p = plot(p1, p2, p3, legend = false, layout = (3, 1))

# display(p)
