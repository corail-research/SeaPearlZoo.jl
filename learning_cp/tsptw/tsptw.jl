using SeaPearl
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux
using Statistics

using Plots
gr()

####
include("rewards.jl")
####

n_city = 51
# n_city = 11
grid_size = 100
max_tw_gap = 10
max_tw = 100

numInFeatures = 6

tsptw_generator = SeaPearl.TsptwGenerator(n_city, grid_size, max_tw_gap, max_tw, true)

# numberOfCPNodes = 1 + floor(Int64, coloring_generator.nb_nodes * ( 3 + coloring_generator.density ))
#numberOfCPNodes = 141

state_size = (n_city, n_city+6+2)

n_episodes = 5001
n_trainings = 20
loss = zeros(n_episodes)
struct TsptwVariableSelection{TakeObjective} <: SeaPearl.AbstractVariableSelection{TakeObjective} end
TsptwVariableSelection(;take_objective=false) = TsptwVariableSelection{take_objective}()
selectMin(x::SeaPearl.IntVar) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)

# for _ in 1:n_trainings
include("agents.jl")
learnedHeuristic = SeaPearl.LearnedHeuristic{SeaPearl.TsptwStateRepresentation, SeaPearl.TsptwReward, SeaPearl.VariableOutput}(agent)


function (::TsptwVariableSelection{false})(cpmodel::SeaPearl.CPModel; rng=nothing)
    for i in 1:length(keys(cpmodel.variables))
        # if haskey(cpmodel.variables, "a_"*string(i))
        #     # println("cpmodel.variables[*string(i)]", cpmodel.variables["a_"*string(i)])
        #     println(cpmodel.variables["a_"*string(i)])
        # end
        if haskey(cpmodel.variables, "a_"*string(i)) && !SeaPearl.isbound(cpmodel.variables["a_"*string(i)])
            # println("cpmodel.variables[*string(i)]", cpmodel.variables["a_"*string(i)])
            return cpmodel.variables["a_"*string(i)]
        end
    end
    println("NO FREE VAR FOUND")
    for i in 1:length(keys(cpmodel.variables))
        if haskey(cpmodel.variables, "a_"*string(i))
            # println("cpmodel.variables[*string(i)]", cpmodel.variables["a_"*string(i)])
            println(cpmodel.variables["a_"*string(i)])
        end
    end
    println(cpmodel.variables)
end

variableSelection = TsptwVariableSelection()


maxNumberOfSteps = n_episodes
loss = zeros(maxNumberOfSteps)
rewards = zeros(maxNumberOfSteps)
numberOfSteps = 1
function metricsFun(;kwargs...)
    if numberOfSteps < maxNumberOfSteps
        loss[numberOfSteps] = kwargs[:loss]
        rewards[numberOfSteps] = kwargs[:total_reward]
        global numberOfSteps += 1
    end
end

############# TRAIN

eval_freq = 250

bestsolutions, nodevisited, timeneeded, eval_nodes, eval_tim = SeaPearl.train!(
    valueSelectionArray=[learnedHeuristic, heuristic_min], 
    generator=tsptw_generator,
    nb_episodes=n_episodes,
    strategy=SeaPearl.DFSearch,
    variableHeuristic=variableSelection,
    out_solver=true,
    metricsFun=metricsFun,
    verbose = false,
    evaluator=SeaPearl.SameInstancesEvaluator(; eval_freq = eval_freq, nb_instances = 5)
)

# ode_visited_delta += (nodevisited[:, 1]-nodevisited[:, 2])/n_trainings
# end

# plot 
# a = size(node_visited_delta, 1)
x = 1:n_episodes
mean_nodevisited = zeros(n_episodes)
mean_reward = zeros(n_episodes)
mean_over = 10
for i in (mean_over+1):n_episodes
    mean_nodevisited[i] = mean(nodevisited[(i-mean_over):i])
    mean_reward[i] = mean(rewards[(i-mean_over):i])
end
p1 = plot(x, [mean_nodevisited .- 2, mean_reward], xlabel="Episode", ylabel="Reward/Steps", ylims = [0, 11])
display(p1)
p5 = plot(1:(floor(Int64, n_episodes/eval_freq)+1), eval_nodes[1:floor(Int64, n_episodes/eval_freq)+1, :], xlabel="Evaluation", ylabel="Number of nodes visited", ylims = [0, 800])
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
