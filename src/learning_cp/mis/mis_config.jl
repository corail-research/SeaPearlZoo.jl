struct MisAgentConfig
    gamma               :: Float32
    batch_size          :: Int
    output_size         :: Int
    update_horizon      :: Int
    min_replay_history  :: Int
    update_freq         :: Int
    target_update_freq  :: Int
    trajectory_capacity :: Int
end

struct MisExperimentSettings
    nbEpisodes          :: Int     
    restartPerInstances :: Int     
    evalFreq            :: Int     
    nbInstances         :: Int     
    nbRandomHeuristics  :: Int     
    nbNewVertices       :: Int     
    nbInitialVertices   :: Int
end