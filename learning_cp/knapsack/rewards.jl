
struct IlanReward <: CPRL.AbstractReward end  

function CPRL.set_reward!(::CPRL.DecisionPhase, lh::CPRL.LearnedHeuristic{SR, IlanReward, O}, model::CPRL.CPModel) where {
    SR <: CPRL.AbstractStateRepresentation,
    O <: CPRL.ActionOutput
}
    lh.current_reward -= 1/40
    nothing  
end  
    
function CPRL.set_reward!(::CPRL.EndingPhase, env::CPRL.LearnedHeuristic{SR, IlanReward, O}, model::CPRL.CPModel, symbol::Union{Nothing, Symbol}) where { 
    SR <: CPRL.AbstractStateRepresentation,
    O <: CPRL.ActionOutput
}
    env.reward += 100/model.statistics.numberOfNodes + 10 
    nothing  
end  
