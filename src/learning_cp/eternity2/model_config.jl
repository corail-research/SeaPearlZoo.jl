using SeaPearl

struct EternityModelConfig
    activation_function
    gpu :: Bool
    num_input_features :: Int    
end

struct EternityAgentConfig
    m :: Int
    n :: Int ## m, n are the dimensions of the eternity2 board
    approximator_model :: SeaPearl.FullFeaturedCPNN
    target_approximator_model :: SeaPearl.FullFeaturedCPNN
end

struct DQNLearnerConfig
    gamma :: Float32
    batch_size :: Int
    update_horizon :: Int
    min_replay_history :: Int
    update_freq :: Int
    target_update_freq :: Int
end