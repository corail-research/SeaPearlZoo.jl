
struct InspectReward <: SeaPearl.AbstractReward end 

function SeaPearl.set_reward!(::SeaPearl.StepPhase, lh::SeaPearl.LearnedHeuristic{SR, InspectReward, O}, model::SeaPearl.CPModel, symbol::Union{Nothing, Symbol}) where {
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    lh.current_reward = 0
    nothing
end

function SeaPearl.set_reward!(::SeaPearl.DecisionPhase, lh::SeaPearl.LearnedHeuristic{SR, InspectReward, O}, model::SeaPearl.CPModel) where {
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    lh.current_reward += -1/40
    nothing
end

function SeaPearl.set_reward!(::SeaPearl.EndingPhase, lh::SeaPearl.LearnedHeuristic{SR, InspectReward, O}, model::SeaPearl.CPModel, symbol::Union{Nothing, Symbol}) where { 
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    lh.current_reward += 10 + 300/(model.statistics.numberOfNodes)
    nothing
end
