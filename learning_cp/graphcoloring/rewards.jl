
struct InspectReward <: SeaPearl.AbstractReward end 

function SeaPearl.set_reward!(::SeaPearl.StepPhase, lh::SeaPearl.LearnedHeuristic{SR, InspectReward, O}, model::SeaPearl.CPModel, symbol::Union{Nothing, Symbol}) where {
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    if !isnothing(lh.search_metrics.current_best) && lh.search_metrics.current_best == 3
        lh.current_reward = 0
    else
        lh.current_reward += 0
        #= 
        if symbol == :Infeasible
            if isempty(model.solutions)
                lh.current_reward += - 5 * (15) * lh.search_metrics.last_unfeasible
            else
                lh.current_reward += - 5 * (lh.search_metrics.current_best - 4) * lh.search_metrics.last_unfeasible
            end
        end

        if symbol == :FoundSolution
            if isempty(model.solutions)
                lh.current_reward += + 50 * (4 - lh.search_metrics.current_best)/ (1 + lh.search_metrics.last_foundsolution)
            else
                lh.current_reward += + 50 * (20)/ (1 + lh.search_metrics.last_foundsolution)
            end
        end =#
    end
    #println("Rewarding phase : ", symbol, "  ", lh.search_metrics, "  ", model.statistics.numberOfNodes)
    nothing
end

function SeaPearl.set_reward!(::SeaPearl.DecisionPhase, lh::SeaPearl.LearnedHeuristic{SR, InspectReward, O}, model::SeaPearl.CPModel) where {
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    if !isnothing(lh.search_metrics.current_best) && lh.search_metrics.current_best == 3
        lh.current_reward = 0
    else
        lh.current_reward += -1
    end
    #lh.current_reward += -1
    #println("Decision phase : ", "  ", lh.search_metrics, "  ", model.statistics.numberOfNodes)
    nothing
end

function SeaPearl.set_reward!(::SeaPearl.EndingPhase, env::SeaPearl.LearnedHeuristic{SR, InspectReward, O}, model::SeaPearl.CPModel, symbol::Union{Nothing, Symbol}) where { 
    SR <: SeaPearl.AbstractStateRepresentation,
    O <: SeaPearl.ActionOutput
}
    #lh.current_reward = - 10 * model.statistics.numberOfNodes
    #lh.current_reward += - 30/(model.statistics.numberOfNodes)
    #lh.current_reward = + lh.search_metrics.total
    nothing
end
