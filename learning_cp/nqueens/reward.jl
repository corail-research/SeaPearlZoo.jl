
struct IlanReward <: SeaPearl.AbstractReward end  

function SeaPearl.set_reward!(::SeaPearl.StepPhase, lh::SeaPearl.LearnedHeuristic{SR,IlanReward,O}, model::SeaPearl.CPModel, symbol::Union{Nothing,Symbol}) where {SR <: SeaPearl.AbstractStateRepresentation,O <: SeaPearl.ActionOutput}
    lh.current_reward -= 1
    nothing
end

function SeaPearl.set_reward!(::SeaPearl.DecisionPhase, lh::SeaPearl.LearnedHeuristic{SR,IlanReward,O}, model::SeaPearl.CPModel) where {SR <: SeaPearl.AbstractStateRepresentation,O <: SeaPearl.ActionOutput}
    lh.current_reward -= 1
    nothing  
end  
    
function SeaPearl.set_reward!(::SeaPearl.EndingPhase, lh::SeaPearl.LearnedHeuristic{SR,IlanReward,O}, model::SeaPearl.CPModel, symbol::Union{Nothing,Symbol}) where {SR <: SeaPearl.AbstractStateRepresentation,O <: SeaPearl.ActionOutput}
    lh.current_reward += 100 / model.statistics.numberOfNodes
    nothing  
end  
