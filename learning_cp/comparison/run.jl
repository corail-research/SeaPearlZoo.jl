include("nqueens.jl")

experiment_nn_heterogeneous_nqueens(5, 101, 1; reward=SeaPearl.CPReward)
experiment_explorer_heterogeneous_nqueens(5, 101, 1; reward=SeaPearl.CPReward)
experiment_chosen_features_heterogeneous_nqueens(5, 101, 1; reward=SeaPearl.CPReward)
experiment_representation_nqueens(5, 101, 1, reward=SeaPearl.CPReward)