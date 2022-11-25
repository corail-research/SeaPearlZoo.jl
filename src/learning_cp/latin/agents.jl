using Flux
using SeaPearl

include("agent_config.jl")

function build_latin_model(model_config::LatinModelConfig)
    model = SeaPearl.CPNN(
        graphChain = Flux.Chain(
            SeaPearl.GraphConv(model_config.num_imput_features => 32, Flux.leakyrelu),
            [model_config.layer for i = 1: model_config.num_layers]...
        ),
        nodeChain = Flux.Chain(
            Flux.Dense(32, 32, Flux.leakyrelu),
            Flux.Dense(32, 32, Flux.leakyrelu),
            Flux.Dense(32, 16, Flux.leakyrelu),
        ),
        globalChain = Flux.Chain(
            Flux.Dense(model_config.num_global_features, 32, Flux.leakyrelu),
            Flux.Dense(32, 32, Flux.leakyrelu),
            Flux.Dense(32, 16, Flux.leakyrelu),
        ),
        outputChain = Flux.Chain(
            Flux.Dense(32, 32, Flux.leakyrelu),
            Flux.Dense(32, model_config.N),
        )
    )
    if model_config.gpu
        return model |> gpu
    end
    return model
end

function build_latin_agent(agent_config::LatinAgentConfig)
    agent = RL.Agent(
        policy = RL.QBasedPolicy(
            learner = RL.DQNLearner(
                approximator = RL.NeuralNetworkApproximator(
                    model = agent_config.approximator,
                    optimizer = ADAM()
                ),
                target_approximator = RL.NeuralNetworkApproximator(
                    model = agent_config.target_approximator,
                    optimizer = ADAM()
                ),
                loss_func = Flux.Losses.huber_loss,
                batch_size = agent_config.batch_size,                
                update_horizon = agent_config.update_horizon,        
                min_replay_history = agent_config.min_replay_history,
                update_freq = agent_config.update_freq,              
                target_update_freq = agent_config.target_update_freq 
            ),
            explorer = RL.EpsilonGreedyExplorer(
                ϵ_stable = agent_config.ϵ_stable,
                decay_steps = agent_config.decay_steps,
                step = agent_config.explorer_step
            )
        ),
        trajectory = RL.CircularArraySLARTTrajectory(
            capacity = agent_config.trajectory_capacity,
            state = SeaPearl.DefaultTrajectoryState[] => (),
            legal_actions_mask = Vector{Bool} => (agent_config.N, ),
        )
    )
    return agent
end
