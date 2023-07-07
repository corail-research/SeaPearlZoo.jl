# Model definition
include("../model_config.jl")

function build_knapsack_actor_model(output_size:: Int)
    approximator_model = SeaPearl.CPNN(
        graphChain = Flux.Chain(
            SeaPearl.GraphConv(numInFeatures => 16, Flux.leakyrelu),
            SeaPearl.GraphConv(16 => 16, Flux.leakyrelu),
            SeaPearl.GraphConv(16 => 16, Flux.leakyrelu)
        ),
        nodeChain = Flux.Chain(
            Flux.Dense(16, 32, Flux.leakyrelu),
            Flux.Dense(32, 16, Flux.leakyrelu),
        ),
        outputChain = Flux.Chain(
            Flux.Dense(16, 16, Flux.leakyrelu),
            Flux.Dense(16, output_size),
        )) #|> gpu
    return approximator_model
end

function build_knapsack_critic_model(output_size:: Int)
    approximator_model = SeaPearl.CPNN(
        graphChain = Flux.Chain(
            SeaPearl.GraphConv(numInFeatures => 16, Flux.leakyrelu),
            SeaPearl.GraphConv(16 => 16, Flux.leakyrelu),
            SeaPearl.GraphConv(16 => 16, Flux.leakyrelu)
        ),
        nodeChain = Flux.Chain(
            Flux.Dense(16, 32, Flux.leakyrelu),
            Flux.Dense(32, 16, Flux.leakyrelu),
        ),
        outputChain = Flux.Chain(
            Flux.Dense(16, 16, Flux.leakyrelu),
            Flux.Dense(16, 1),
        )) #|> gpu
    return approximator_model
end


function build_knapsack_ppo_agent(actor_model, critic_model, agent_config::KnapsackPPOAgentConfig)
    agent = RL.Agent(
        policy = RL.PPOPolicy(
            approximator = RL.ActorCritic(
                actor = RL.NeuralNetworkApproximator(
                    model = actor_model,
                    optimizer = ADAM(),
                ),
                critic = RL.NeuralNetworkApproximator(
                    model = critic_model,
                    optimizer = ADAM(),
                ),
                optimizer = ADAM(),
            ) |> gpu,
            γ = agent_config.gamma,
            λ = agent_config.lambda,
            actor_loss_weight = agent_config.actor_loss_weight,
            critic_loss_weight = agent_config.critic_loss_weight,
            entropy_loss_weight = agent_config.entropy_loss_weight,
            update_freq = agent_config.update_freq,
            clip_range = agent_config.clip_range,
            max_grad_norm = agent_config.max_grad_norm,
            n_epochs = agent_config.n_epochs,
            n_microbatches = agent_config.n_microbatches,
        ),
        trajectory = RL.MaskedPPOTrajectory(;
            capacity = agent_config.trajectory_capacity,
            action_log_prob = Vector{Float32} => (),
            state = SeaPearl.DefaultTrajectoryState[] => (),
            legal_actions_mask = Vector{Bool} => (agent_config.output_size, ),
            terminal = Bool => (1,),
            reward = Float32 => (1,),
        ),
    )
    return agent
end
