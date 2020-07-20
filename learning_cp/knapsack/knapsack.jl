using CPRL
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using Statistics

using Plots
gr()

include("rewards.jl")

knapsack_generator = CPRL.KnapsackGenerator(12, 10, 0.2)

numberOfFeatures = 10
numInFeatures = numberOfFeatures
include("features.jl")


maxNumberOfCPnodes = 300
state_size = (maxNumberOfCPnodes, numInFeatures + maxNumberOfCPnodes + 3, 1) 
println("state_size", state_size)

include("agents.jl")

learnedHeuristic = CPRL.LearnedHeuristic{CPRL.DefaultStateRepresentation, IlanReward, CPRL.VariableOutput}(agent, maxNumberOfCPnodes)

basicHeuristic = CPRL.BasicHeuristic((x) -> CPRL.maximum(x.domain))

function selectNonObjVariable(model::CPRL.CPModel)
    selectedVar = nothing
    minSize = typemax(Int)
    for (k, x) in model.variables
        if length(x.domain) > 1 && length(x.domain) < minSize# && k != "numberOfColors"
            selectedVar = x
            minSize = length(x.domain)
        end
    end
    # @assert !isnothing(selectedVar)
    if isnothing(selectedVar)
        return model.variables["numberOfColors"]
    end
    return selectedVar
end

maxNumOfEpisodes = 4000

meanNodeVisited = Array{Float32}(undef, maxNumOfEpisodes)
meanNodeVisitedBasic = Array{Float32}(undef, maxNumOfEpisodes)
nodeVisitedBasic = Array{Int64}(undef, maxNumOfEpisodes)
nodeVisitedLearned = Array{Int64}(undef, maxNumOfEpisodes)

meanOver = 50
sum = 0
function metricsFun(;kwargs...)
    i = kwargs[:episode]
    if kwargs[:heuristic] == learnedHeuristic
        currentNodeVisited = kwargs[:nodeVisited]
        nodeVisitedLearned[i] = currentNodeVisited

        currentMean = 0.
        if i <= meanOver
            currentMean = mean(nodeVisitedLearned[1:i])
        else
            currentMean = mean(nodeVisitedLearned[(i-meanOver+1):i])
        end
        meanNodeVisited[i] = currentMean
    else
        currentNodeVisited = kwargs[:nodeVisited]
        nodeVisitedBasic[i] = currentNodeVisited

        currentMean = 0.
        if i <= meanOver
            currentMean = mean(nodeVisitedBasic[1:i])
        else
            currentMean = mean(nodeVisitedBasic[(i-meanOver+1):i])
        end
        meanNodeVisitedBasic[i] = currentMean
    end
end



function trytrain(nepisodes::Int)
    
    bestsolutions, nodevisited = CPRL.train!(
        valueSelectionArray=[learnedHeuristic, basicHeuristic], 
        generator=knapsack_generator,
        nb_episodes=nepisodes,
        strategy=CPRL.DFSearch,
        variableHeuristic=selectNonObjVariable,
        metricsFun=metricsFun,
        verbose=false
    )
    # println(bestsolutions)
    # nodevisited = Array{Any}([35, 51])
    # nodevisited = convert(Array{Int}, nodevisited)
    # println(nodevisited)

    
    # plot 
    x = 1:nepisodes

    p = plot(x, 
            [nodeVisitedLearned[1:nepisodes] meanNodeVisited[1:nepisodes] nodeVisitedBasic[1:nepisodes] meanNodeVisitedBasic[1:nepisodes] (nodeVisitedLearned-nodeVisitedBasic)[1:nepisodes] (meanNodeVisited-meanNodeVisitedBasic)[1:nepisodes]], 
            xlabel="Episode", 
            ylabel="Number of nodes visited", 
            label = ["Learned" "mean/$meanOver Learned" "Basic" "mean/$meanOver Basic" "Delta" "Mean Delta"],
            ylims = (-50,300)
            )
    display(p)
    return nodeVisitedLearned, meanNodeVisited, nodeVisitedBasic
end


