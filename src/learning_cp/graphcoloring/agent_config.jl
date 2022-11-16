struct AgentConfig
    gamma :: Float32
    batch_size :: Int
    update_horizon :: Int
    min_replay_history :: Int
    update_freq :: Int
    target_update_freq :: Int
    trajectory_capacity :: Int
end