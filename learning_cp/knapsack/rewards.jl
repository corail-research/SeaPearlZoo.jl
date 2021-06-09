
mutable struct knapsackReward <: SeaPearl.AbstractReward  
    value::Float32
end

knapsackReward(model::SeaPearl.CPModel) = knapsackReward(0)

function SeaPearl.set_reward!(::Type{SeaPearl.StepPhase}, lh::SeaPearl.LearnedHeuristic{SR, knapsackReward, O}, model::SeaPearl.CPModel, symbol::Union{Nothing, Symbol}) where {
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    lh.reward.value += -1
    nothing
end

function SeaPearl.set_reward!(::Type{SeaPearl.DecisionPhase}, lh::SeaPearl.LearnedHeuristic{SR, knapsackReward, O}, model::SeaPearl.CPModel) where {
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    lh.reward.value += -1
    nothing  
end  
    
function SeaPearl.set_reward!(::Type{SeaPearl.EndingPhase}, lh::SeaPearl.LearnedHeuristic{SR, knapsackReward, O}, model::SeaPearl.CPModel, symbol::Union{Nothing, Symbol}) where { 
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    nothing  
end  
