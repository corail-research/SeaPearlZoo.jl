struct ColoringAgentConfig
    gamma               :: Float32
    batch_size          :: Int
    output_size         :: Int
    update_horizon      :: Int
    min_replay_history  :: Int
    update_freq         :: Int
    target_update_freq  :: Int
    trajectory_capacity :: Int
end

struct ColoringExperimentSettings
    nbEpisodes          :: Int     
    restartPerInstances :: Int     
    evalFreq            :: Int     
    nbInstances         :: Int     
    nbRandomHeuristics  :: Int     
    nbNodes             :: Int     
    nbMinColor          :: Int     
    density             :: Float32
end

struct ColoringPPOAgentConfig
    gamma               :: Float32
    lambda              :: Float32
    clip_range          :: Float32
    max_grad_norm       :: Float32
    n_epochs            :: Int
    n_microbatches      :: Int
    actor_loss_weight   :: Float32
    critic_loss_weight  :: Float32
    entropy_loss_weight :: Float32
    output_size         :: Int
    update_freq         :: Int
    trajectory_capacity :: Int
end