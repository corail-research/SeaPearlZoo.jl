include("nqueens.jl")
include("graphcoloring.jl")

# experiment_nn_heterogeneous_nqueens(25, 20001, 1; reward=SeaPearl.GeneralReward)
# experiment_explorer_heterogeneous_nqueens(20, 10001, 10)
# experiment_chosen_features_heterogeneous_nqueens(10, 1001, 10; )
# experiment_pooling_heterogeneous_graphcoloring(10, 5, 0.95, 3001, 10)
# experiment_chosen_features_heterogeneous_graphcoloring(10, 5, 0.95, 3001, 10)
experiment_nn_heterogeneous_graphcoloring(30, 5, 0.95, 3001, 10)
