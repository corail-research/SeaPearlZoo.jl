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

function save_experiment_artefacts(
    experiment_parameters::Dict,
    experiment_config::TSPTWExperimentConfig,
    agent::RL.Agent
)
    experiment_time = now()
    dir = mkdir(string("exp_",Base.replace("$(round(experiment_time, Dates.Second(3)))",":"=>"-")))
    open(dir*"/params.json", "w") do file
        JSON.print(file, experiment_parameters)
    end
    trained_weights = params(agent.policy.learner.approximator.model)
    @save dir*"/model_weights_tsptw"*string(experiment_config.num_cities)*".bson" trained_weights
end

function get_experiment_parameters(
    experiment_config::TSPTWExperimentConfig, 
    agent::RL.Agent, 
    learned_heuristic::SeaPearl.SimpleLearnedHeuristic
)
    experiment_parameters = Dict(
        :experimentParameters => Dict(
            :experiment_config.num_episodes => experiment_config.num_episodes,
            :experiment_config.num_restarts_per_instance => experiment_config.num_restarts_per_instance,
            :experiment_config.eval_freq => experiment_config.eval_freq,
            :experiment_config.num_instances => experiment_config.num_instances
        ),
        :generatorParameters => Dict(
            :instance => "tsptw",
            :num_cities => experiment_config.num_cities,
            :experiment_config.grid_size => experiment_config.grid_size,
            :experiment_config.max_tw_gap => experiment_config.max_tw_gap,
            :experiment_config.max_tw => experiment_config.max_tw
        ),
        :learned_heuristic => Dict(
            :learned_heuristic_type => typeof(learned_heuristic),
            :eta_init => hasproperty(learned_heuristic, :eta_init) ? learned_heuristic.eta_init : nothing,
            :eta_stable => hasproperty(learned_heuristic, :eta_stable) ? learned_heuristic.eta_stable : nothing,
            :warmup_steps => hasproperty(learned_heuristic, :warmup_steps) ? learned_heuristic.warmup_steps : nothing,
            :decay_steps => hasproperty(learned_heuristic, :decay_steps) ? learned_heuristic.decay_steps : nothing,
            :rng => hasproperty(learned_heuristic, :rng) ? Dict(:rngType => typeof(learned_heuristic.rng), :seed => learned_heuristic.rng.seed) : nothing
        ),
        :nbrandom_heuristics => nbrandom_heuristics,
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