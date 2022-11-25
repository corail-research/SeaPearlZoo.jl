using SeaPearl

struct LatinExperimentConfig
    state_representation# :: SeaPearl.DefaultStateRepresentation{LatinFeaturization, SeaPearl.DefaultTrajectoryState}
    num_episodes :: Int
    eval_freq :: Int
    num_instances :: Int
    num_restarts_per_instance :: Int
    num_random_heuristics :: Int
    N :: Int
    p :: Float64
    generator :: SeaPearl.LatinGenerator
    function LatinExperimentConfig(
            state_representation,#::SeaPearl.DefaultStateRepresentation{LatinFeaturization, SeaPearl.DefaultTrajectoryState}, 
            num_episodes::Int, 
            eval_freq::Int, 
            num_instances::Int,
            num_restarts_per_instance :: Int,
            num_random_heuristics::Int, 
            N::Int, 
            p::Float64
        )
        return new(
            state_representation, 
            num_episodes, 
            eval_freq, 
            num_instances,
            num_restarts_per_instance,
            num_random_heuristics,
            N, 
            p, 
            SeaPearl.LatinGenerator(N, p)
        )
    end
end