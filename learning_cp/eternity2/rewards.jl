#An inspect Reward to find a solution in a few number of steps, in the case when one episode corresponds to finind a solution.

mutable struct InspectReward <: SeaPearl.AbstractReward
    value::Float32
end

InspectReward(model::SeaPearl.CPModel) = InspectReward(0)

function SeaPearl.set_reward!(::Type{SeaPearl.StepPhase}, lh::SeaPearl.LearnedHeuristic{SR, InspectReward, O}, model::SeaPearl.CPModel, symbol::Union{Nothing, Symbol}) where {
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    if symbol == :Infeasible
        lh.reward.value -= 10
    elseif symbol == :Feasible
        lh.reward.value -= 1
    end
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
    lh.reward.value += 200
    nothing
end
