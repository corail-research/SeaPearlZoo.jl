using BSON: @save, @load
using Dates
using JSON
using ReinforcementLearning
const RL = ReinforcementLearning
using SeaPearl

include("coloring_config.jl")

function solve_learning_coloring(
    agent::RL.Agent,
    agent_config::ColoringAgentConfig,
    coloring_settings::ColoringExperimentSettings,
    instance_generator::SeaPearl.AbstractModelGenerator,
    eval_generator= nothing,
    save_experiment_parameters::Bool = false,
    save_model::Bool = false
    )
    
    if save_experiment_parameters
        experiment_time = now()
        dir = mkdir(string("exp_",Base.replace("$(round(experiment_time, Dates.Second(3)))",":" => "-")))
        experiment_parameters = get_experiment_parameters(agent, agent_config, coloring_settings)
        open(dir*"/params.json", "w") do file
            JSON.print(file, experiment_parameters)
        end
    end

    if !isnothing(eval_generator)
        evaluator = SeaPearl.SameInstancesEvaluator(valueSelectionArray, eval_generator; evalFreq=coloring_settings.evalFreq, nbInstances=coloring_settings.nbInstances, evalTimeOut = coloring_settings.evalTimeOut, rng = MersenneTwister(coloring_settings.seedEval))
    else
        evaluator = SeaPearl.SameInstancesEvaluator(valueSelectionArray, instance_generator; evalFreq=coloring_settings.evalFreq, nbInstances=coloring_settings.nbInstances, evalTimeOut = coloring_settings.evalTimeOut, rng = MersenneTwister(coloring_settings.seedEval))
    end

    metricsArray, eval_metricsArray = SeaPearl.train!(
        valueSelectionArray=valueSelectionArray,
        generator=instance_generator,
        nbEpisodes=coloring_settings.nbEpisodes ,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variableSelection,
        out_solver=true,
        verbose = false,
        evaluator=evaluator,
        restartPerInstances = coloring_settings.restartPerInstances
    )

    folder_path = "saved_model/gc"

    if save_model
        if !isdir(folder_path)
            mkdir(folder_path)
            println(" Folder created successfully!")
        else
            println(" Folder already exists!")
        end
    end

    if save_model 
        if (hasfield(typeof(agent.policy),:approximator)) # PPO
            model = agent.policy.approximator
        else # DQN
            model = agent.policy.learner.approximator
        end
        @save folder_path*"/model_gc"*string(instance_generator.n)*"_"*string(coloring_settings.nbNodes)*"_"*string(coloring_settings.nbNodesEval)*"_"*string(coloring_settings.nbEpisodes)*".bson" model
    end

    return metricsArray, eval_metricsArray
end

function get_experiment_parameters(agent::RL.Agent, agent_config::ColoringAgentConfig, coloring_settings::ColoringExperimentSettings)
    experiment_parameters = Dict(
        :experimentParameters => Dict(
            :nbEpisodes => coloring_settings.nbEpisodes ,
            :restartPerInstances => coloring_settings.restartPerInstances,
            :evalFreq => coloring_settings.evalFreq,
            :nbInstances => coloring_settings.nbInstances,
        ),
        :generatorParameters => Dict(
            :nbNodes => coloring_settings.nbNodes,
            :nbMinColor => coloring_settings.nbMinColor,
            :density => coloring_settings.density
        ),
        :nbRandomHeuristics => coloring_settings.nbRandomHeuristics,
        :Featurization => Dict(
            :featurizationType => SeaPearl.AbstractFeaturization,
            :chosen_features => nothing
        ),
        :learnerParameters => Dict(
            :model => string(agent.policy.learner.approximator.model),
            :gamma => agent.policy.learner.sampler.γ,
            :batch_size => agent.policy.learner.sampler.batch_size,
            :update_horizon => agent.policy.learner.sampler.n,
            :min_replay_history => agent.policy.learner.min_replay_history,
            :update_freq => agent.policy.learner.update_freq,
            :target_update_freq => agent.policy.learner.target_update_freq,
        ),
        :explorerParameters => Dict(
            :ϵ_stable => agent.policy.explorer.ϵ_stable,
            :decay_steps => agent.policy.explorer.decay_steps,
        ),
        :trajectoryParameters => Dict(
            :trajectoryType => typeof(agent.trajectory),
            :capacity => agent_config.trajectory_capacity
        ),
        :reward => rewardType
    )
    return experiment_parameters
end 
