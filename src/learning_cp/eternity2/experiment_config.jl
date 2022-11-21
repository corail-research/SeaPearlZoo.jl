using SeaPearl

struct ExperimentConfig
    instance_generator :: SeaPearl.AbstractModelGenerator
    num_episodes :: Int
    eval_freq :: Int
    num_instances :: Int
    num_random_heuristics :: Int
end