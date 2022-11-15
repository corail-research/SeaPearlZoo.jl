struct AgentConfig
    gamma :: Int   # = 0.99f0,
    batch_size :: Int # = 16,
    update_horizon :: Int # = 4
    min_replay_history :: Int # = 128,
    update_freq :: Int # = 1,
    target_update_freq :: Int # = 200,
    trajectory_capacity :: Int # = 3000
end