using SeaPearl

mutable struct KepParameters
    num_nodes             ::Int
    num_nodes_eval         ::Int
    density             ::Float64
    num_episodes          ::Int 
    num_restarts_per_instance  ::Int
    num_evals             ::Int
    num_instances         ::Int
    num_random_heuristics  ::Int
    eval_timeout         ::Int
    batch_size           ::Int
    seed                ::Union{Nothing,Int}
    seed_eval            ::Union{Nothing,Int}
    reward              ::Type{<:SeaPearl.AbstractReward}
    strategy            ::SeaPearl.SearchStrategy
    eval_strategy        ::SeaPearl.SearchStrategy
    SR                  ::Union{Nothing,Type{<:SeaPearl.AbstractStateRepresentation}}
    num_input_features       ::Union{Nothing,Int}

    # Experiment default values
    function KepParameters()
        return new(
            10,                        # num_nodes
            10,                        # num_nodes_eval
            0.2,                       # density
            500,                       # num_episodes #usually 1000
            20,                        # num_restarts_per_instance
            20,                        # num_evals
            10,                        # num_instances
            1,                         # num_random_heuristics #usually 5
            300,                       # eval_timeout
            1,                         # batch_size
            nothing,                   # seed
            nothing,                   # seed_eval
            SeaPearl.GeneralReward,    # reward
            SeaPearl.DFSearch(),       # strategy
            SeaPearl.DFSearch(),       # eval_strategy
            nothing,                   # SR - state representation
            nothing                    # num_input_features
        )
    end
end