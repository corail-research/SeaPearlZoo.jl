struct ModelConfig
    num_input_features :: Int
    board_size :: Int
    gpu :: Bool
end

struct AgentConfig
    approximator_model :: SeaPearl.CPNN
    target_approximator_model :: SeaPearl.CPNN
    batch_size :: Int
    update_horizon :: Int
    min_replay_history :: Int
    update_freq :: Int
    target_update_freq :: Int
    ϵ_init :: Float64
    ϵ_stable :: Float64
    kind
    decay_steps :: Int
    step :: Int
    trajectory_capacity :: Int
    board_size :: Int
end