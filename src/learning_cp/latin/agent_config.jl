using SeaPearl

include("featurization.jl")

struct LatinModelConfig
    layer :: SeaPearl.GraphConv
    num_layers :: Int16
    num_imput_features :: Int64
    num_global_features :: Int64
    N :: Int
    p :: Float64
    gpu :: Bool
end

struct LatinAgentConfig
    N :: Int
    approximator :: SeaPearl.CPNN
    target_approximator :: SeaPearl.CPNN
    batch_size :: Int
    update_horizon :: Int
    min_replay_history :: Int
    update_freq :: Int
    target_update_freq :: Int
    ϵ_stable :: Float32
    decay_steps :: Int
    explorer_step :: Int
    trajectory_capacity :: Int
end