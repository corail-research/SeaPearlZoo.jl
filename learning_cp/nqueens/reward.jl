
struct nqueenReward <: SeaPearl.AbstractReward end  

function SeaPearl.set_reward!(::Type{SeaPearl.StepPhase}, lh::SeaPearl.LearnedHeuristic{SR,nqueenReward,O}, model::SeaPearl.CPModel, symbol::Union{Nothing,Symbol}) where {SR <: SeaPearl.AbstractStateRepresentation,O <: SeaPearl.ActionOutput}
    lh.current_reward -= 1
    nothing
end

function SeaPearl.set_reward!(::Type{SeaPearl.DecisionPhase}, lh::SeaPearl.LearnedHeuristic{SR,nqueenReward,O}, model::SeaPearl.CPModel) where {SR <: SeaPearl.AbstractStateRepresentation,O <: SeaPearl.ActionOutput}
    lh.current_reward -= 1
    nothing  
end  
    
function SeaPearl.set_reward!(::Type{SeaPearl.EndingPhase}, lh::SeaPearl.LearnedHeuristic{SR,nqueenReward,O}, model::SeaPearl.CPModel, symbol::Union{Nothing,Symbol}) where {SR <: SeaPearl.AbstractStateRepresentation,O <: SeaPearl.ActionOutput}
    lh.current_reward += 100 / model.statistics.numberOfNodes
    nothing  
end  