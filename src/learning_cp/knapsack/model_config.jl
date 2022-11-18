using SeaPearl

struct KnapsackApproximatorConfig
    num_input_features :: Int
    num_GNN_layers :: Int
    GNN_layer :: SeaPearl.GraphConv
    gpu :: Bool
end

struct KnapsackAgentConfig
    gamma :: Float32
    batch_size :: Int
    update_horizon :: Int
    min_replay_history :: Int
    update_freq :: Int
    target_update_freq :: Int
    num_episodes :: Int
end