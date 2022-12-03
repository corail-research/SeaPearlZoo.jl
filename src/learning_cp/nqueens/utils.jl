include("nqueens_config.jl")


function select_random_value(x::SeaPearl.IntVar; cpmodel=nothing)
    selected_number = rand(1:length(x.domain))
    i = 1
    for value in x.domain
        if i == selected_number
            return value
        end
        i += 1
    end
    @assert false "This should not happen"
end

function build_experiment_parameters_dict(
        experiment_config::NQueensConfig,
        learned_heuristic::SeaPearl.SupervisedLearnedHeuristic,
        agent::RL.Agent
    )
    experiment_parameters = Dict(
        :experimentParameters => Dict(
            :nbEpisodes => experiment_config.num_episodes,
            :restartPerInstances => experiment_config.num_restarts_per_instance,
            :evalFreq => experiment_config.eval_freq,
            :nbInstances => experiment_config.num_instances
        ),
        :generatorParameters => Dict(
            :instance => "nqueens",
            :boardSize => board_size,
        ),
        :learnedHeuristic => Dict(
            :learnedHeuristicType => typeof(learned_heuristic),
            :eta_init => hasproperty(learned_heuristic, :eta_init) ? learned_heuristic.eta_init : nothing,
            :eta_stable => hasproperty(learned_heuristic, :eta_stable) ? learned_heuristic.eta_stable : nothing,
            :warmup_steps => hasproperty(learned_heuristic, :warmup_steps) ? learned_heuristic.warmup_steps : nothing,
            :decay_steps => hasproperty(learned_heuristic, :decay_steps) ? learned_heuristic.decay_steps : nothing
        ),
        :nbRandomHeuristics => experiment_config.num_random_heuristics,
        :Featurization => Dict(
            :featurizationType => features_type,
            :chosen_features => nothing
        ),
        :learnerParameters => Dict(
            :model => string(agent.policy.learner.approximator.model),
            :gamma => agent.policy.learner.sampler.γ,
            :batch_size => agent.policy.learner.sampler.batch_size,
            :update_horizon => agent.policy.learner.sampler.n,
            :min_replay_history => agent.policy.learner.min_replay_history,
            :update_freq => agent.policy.learner.update_freq,
            :target_update_freq => agent.policy.learner.target_update_freq
        ),
        :explorerParameters => Dict(
            :ϵ_stable => agent.policy.explorer.ϵ_stable,
            :decay_steps => agent.policy.explorer.decay_steps
        ),
        :trajectoryParameters => Dict(
            :trajectoryType => typeof(agent.trajectory),
            :capacity => trajectory_capacity
        ),
        :reward => reward_type
    )
    return experiment_parameters
end