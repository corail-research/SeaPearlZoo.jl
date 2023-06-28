struct ColoringAgentConfig
    gamma :: Float32
    batch_size :: Int
    output_size :: Int
    update_horizon :: Int
    min_replay_history :: Int
    update_freq :: Int
    target_update_freq :: Int
    trajectory_capacity :: Int
end

struct ColoringExperimentSettings
    nbEpisodes          :: Int     
    restartPerInstances :: Int     
    evalFreq            :: Int     
    evalTimeOut         :: Int
    seedEval            :: Int
    nbInstances         :: Int     
    nbRandomHeuristics  :: Int     
    nbNodes             :: Int   
    nbNodesEval         :: Int  
    nbMinColor          :: Int     
    density             :: Float32
end