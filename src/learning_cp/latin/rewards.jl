using SeaPearl

mutable struct InspectLatinReward <: SeaPearl.AbstractReward
    value::Float32
end

InspectLatinReward(model::SeaPearl.CPModel) = InspectLatinReward(0)

function SeaPearl.set_reward!(::Type{SeaPearl.StepPhase}, lh::SeaPearl.LearnedHeuristic{SR, InspectLatinReward, O}, model::SeaPearl.CPModel, symbol::Union{Nothing, Symbol}) where {
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    lh.reward.value += -1
end

function SeaPearl.set_reward!(::Type{SeaPearl.DecisionPhase}, lh::SeaPearl.LearnedHeuristic{SR, InspectLatinReward, O}, model::SeaPearl.CPModel) where {
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    lh.reward.value += -1
end

function SeaPearl.set_reward!(::Type{SeaPearl.EndingPhase}, lh::SeaPearl.LearnedHeuristic{SR, InspectLatinReward, O}, model::SeaPearl.CPModel, symbol::Union{Nothing, Symbol}) where {
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    lh.reward.value += 100
end
