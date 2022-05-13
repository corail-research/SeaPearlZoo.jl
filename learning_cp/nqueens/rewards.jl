####################
# InspectReward
####################

mutable struct InspectReward <: SeaPearl.AbstractReward
    value::Float32
end

InspectReward(model::SeaPearl.CPModel) = InspectReward(0)

function SeaPearl.set_reward!(::Type{SeaPearl.StepPhase}, lh::SeaPearl.LearnedHeuristic{SR, InspectReward, O}, model::SeaPearl.CPModel, symbol::Union{Nothing, Symbol}) where {
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    lh.reward.value += -1
    nothing
end

function SeaPearl.set_reward!(::Type{SeaPearl.DecisionPhase}, lh::SeaPearl.LearnedHeuristic{SR, InspectReward, O}, model::SeaPearl.CPModel) where {
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    lh.reward.value += -1
    nothing
end

function SeaPearl.set_reward!(::Type{SeaPearl.EndingPhase}, lh::SeaPearl.LearnedHeuristic{SR, InspectReward, O}, model::SeaPearl.CPModel, symbol::Union{Nothing, Symbol}) where {
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    lh.reward.value += 10
    nothing
end

####################
# NQueenReward
####################

struct NQueenReward <: SeaPearl.AbstractReward end  

function SeaPearl.set_reward!(::Type{SeaPearl.StepPhase}, lh::SeaPearl.LearnedHeuristic{SR,NQueenReward,O}, model::SeaPearl.CPModel, symbol::Union{Nothing,Symbol}) where {SR <: SeaPearl.AbstractStateRepresentation,O <: SeaPearl.ActionOutput}
    lh.current_reward -= 1
    nothing
end

function SeaPearl.set_reward!(::Type{SeaPearl.DecisionPhase}, lh::SeaPearl.LearnedHeuristic{SR,NQueenReward,O}, model::SeaPearl.CPModel) where {SR <: SeaPearl.AbstractStateRepresentation,O <: SeaPearl.ActionOutput}
    lh.current_reward -= 1
    nothing  
end  
    
function SeaPearl.set_reward!(::Type{SeaPearl.EndingPhase}, lh::SeaPearl.LearnedHeuristic{SR,NQueenReward,O}, model::SeaPearl.CPModel, symbol::Union{Nothing,Symbol}) where {SR <: SeaPearl.AbstractStateRepresentation,O <: SeaPearl.ActionOutput}
    lh.current_reward += 100 / model.statistics.numberOfNodes
    # Getting the number of conflicts on nqueens(20)
    # Used for debugging only
    # Getting assigned variables
    println("nqueens_conflict_counter")
    boundvariables = Dict{Int,Int}()
    for (id, x) in model.variables
        if isbound(x)
            boundvariables[parse(Int,split(id,"_")[end])] = assignedValue(x)
        end
    end
    # Getting diagonals ids
    posdiags = zeros(39)
    negdiags = zeros(39)
    for (i, j) in boundvariables
        posdiags[i+j-1] += 1
        negdiags[i+(20-j)] += 1
    end

    nb_conflicts = 0
    nb_conflicts += sum(values(counter(values(boundvariables))) .- 1)
    nb_conflicts += sum((posdiags .> 1) .* (posdiags .- 1))
    nb_conflicts += sum((negdiags .> 1) .* (negdiags .- 1))
    lh.reward.value -= nb_conflicts
    nothing  
end  