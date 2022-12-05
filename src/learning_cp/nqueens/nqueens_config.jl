using SeaPearl

struct NQueensLearnedHeuristicConfig
    eta_init :: Float64
    eta_stable :: Float64
    warmup_steps :: Int
    decay_steps :: Int
end

struct NQueensConfig
    board_size :: Int
    generator::SeaPearl.NQueensGenerator
    num_input_features::Int
    num_episodes::Int
    eval_freq::Int
    num_instances::Int
    num_random_heuristics::Int
    num_restarts_per_instance::Int
    reward_type
    save_experiment_artefacts::Bool

    function NQueensConfig(
        board_size::Int,
        num_input_features::Int,
        num_episodes::Int,
        eval_freq::Int,
        num_instances::Int,
        num_random_heuristics::Int,
        num_restarts_per_instance::Int,
        reward_type,
        save_experiment_artefacts::Bool
    )
        return new(
            board_size,
            SeaPearl.NQueensGenerator(board_size),
            num_input_features,
            num_episodes,
            eval_freq,
            num_instances,
            num_random_heuristics,
            num_restarts_per_instance,
            reward_type,
            save_experiment_artefacts
        )
    end
end