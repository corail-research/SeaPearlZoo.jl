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
    nothing  
end  