#include("nqueens.jl")
#include("graphcoloring.jl")
#include("latin.jl")
#include("kep.jl")
include("MIS.jl")
include("MaxCut.jl")
using TensorBoardLogger, Logging, CUDA

# experiment_nn_heterogeneous_nqueens(25, 20001, 1; reward=SeaPearl.GeneralReward)
# experiment_explorer_heterogeneous_nqueens(20, 10001, 10)
# experiment_chosen_features_heterogeneous_nqueens(10, 1001, 10; )
# experiment_pooling_heterogeneous_graphcoloring(10, 5, 0.95, 3001, 10)
# experiment_chosen_features_heterogeneous_graphcoloring(10, 5, 0.95, 3001, 10)
#experiment_nn_heterogeneous_graphcoloring(30, 5, 0.95, 3001, 10)
#experiment_heterogeneous_n_conv_latin(20,0.2,10,10)
#experiment_default_chosen_n_conv_latin(20,0.2,10,10)
#experiment_default_default_n_conv_latin(20,0.2,10,10)
#experiment_chosen_features_heterogeneous_latin(20,0.2,10,100)

#experiment_nn_heterogeneous_graphcoloring(30, 5, 0.9, 500, 10, 30; n_eval=5, pool = SeaPearl.meanPooling())
#experiment_nn_heterogeneous_graphcoloring(30, 5, 0.9, 500, 10, 30; n_eval=5, pool = SeaPearl.sumPooling())
#experiment_nn_heterogeneous_graphcoloring(30, 5, 0.9, 500, 10, 30; n_eval=5, pool = SeaPearl.maxPooling())
#experiment_nn_heterogeneous_latin(10,0.1,2000,10)

#experiment_nn_heterogeneous_kep(20,0.1,1000,10; n_eval=20, pool = SeaPearl.meanPooling()) #jusqu'à 40,, 0.6
#experiment_nn_heterogeneous_kep(20,0.1,1000,10; n_eval=20, pool = SeaPearl.sumPooling())
#experiment_nn_heterogeneous_kep(10,0.8,3000,10; n_eval=20, pool = SeaPearl.maxPooling())

#experiment_different_reward_jobshop(3,10,50,3000,10; n_layers_graph=3, n_eval=10, pool = SeaPearl.meanPooling())

#experiment_different_reward_graph_coloring(30, 5, 0.9, 5000, 10; n_eval=5, pool = SeaPearl.meanPooling())

#simple_experiment_MIS(20,4,100,10; n_eva = 150, k_eva =4, seedEval = 113)
#simple_experiment_MIS(20,4,100,10; n_eva = 150, k_eva =6, seedEval = 123)
#simple_experiment_MIS(40,4,100,10; n_eva = 150, k_eva =6, seedEval = 123)
#simple_experiment_MIS(40,6,100,10; n_eva = 150, k_eva =6, seedEval = 123)
#simple_experiment_MIS(20,4,100,10; n_eva = 150, k_eva =6, seedEval = 123)
#simple_graph_coloring_experiment(10, 50, 5, 0.95, 200, 20; eval_timeout = 60, restartPerInstances = 10, seedEval = 122)
#simple_graph_coloring_experiment(20, 50, 5, 0.95, 200, 20; eval_timeout = 60, restartPerInstances = 10, seedEval = 122)
#simple_graph_coloring_experiment(30, 50, 5, 0.95, 200, 20; eval_timeout = 60, restartPerInstances = 10, seedEval = 122)
#simple_graph_coloring_experiment(40, 50, 5, 0.95, 200, 20; eval_timeout = 60, restartPerInstances = 10, seedEval = 122)
#simple_graph_coloring_experiment(50, 50, 5, 0.95, 200, 20; eval_timeout = 60, restartPerInstances = 10, seedEval = 122)
#simple_experiment_MIS(15,4,20,10; n_eva = 15, k_eva =4, restartPerInstances=1, n_eval=10)
#simple_experiment_kep(20, 0.97, 500, 1, restartPerInstances = 1, n_eval = 1)

#transfert_graph_coloring_experiment(10, 10, 5, 0.95, 500, 20; restartPerInstances = 10, seedEval = 165)

#simple_experiment_MIS(40,4,600,10; n_eva = 40, k_eva =4, seedEval = 163, n_eval=30)
#simple_experiment_MIS(40,6,800,10; n_eva = 40, k_eva =6, seedEval = 163, n_eval=30)

#simple_experiment_MaxCut(20, 4, 100, 10; restartPerInstances = 50 )
#simple_experiment_MaxCut(20, 4, 600, 10; restartPerInstances = 10, n_eva = 20, k_eva = 4, seedEval = 134, pool = SeaPearl.maxPooling())

#simple_MaxCut_cpnn(20, 4, 600, 10; restartPerInstances = 10, n_eva = 20, k_eva = 4, seedEval = 134)
#experiment_MIS_dfs_dive(30,5,35000,1, seedEval = 124, restartPerInstances = 1, training_timeout =7200, eval_every=240)

#simple_experiment_MaxCut(80, 6, 10000, 20; reward = SeaPearl.ScoreReward,  eval_timeout = 120, restartPerInstances = 1, seedEval = 137, pool = SeaPearl.meanPooling(), eval_strategy = SeaPearl.ILDSearch(2), device = gpu, batch_size = 264)


#ARGS =["maxwell01","030"]

#simple_experiment_MIS(30,6,30000,5; restartPerInstances = 1, seedEval = 161, reward = SeaPearl.GeneralReward, pool = SeaPearl.sumPooling(), device = gpu, batch_size= 32, update_freq = 1, target_update_freq = 100, name = "MIS_final_benchmark_S_sumPooling_RFS", numDevice = 0, eval_strategy = SeaPearl.DFSearch(), single_dive = true, training_timeout = 40000)



simple_experiment_MIS(30,6,7500,1; restartPerInstances = 1, seedEval = 148, reward = SeaPearl.DefaultReward, pool = SeaPearl.sumPooling(), device = gpu, batch_size= 32, update_freq = 1, target_update_freq = 100, name = "105_MIS_final_benchmark_S_sumPooling_DFS_40000sec", numDevice = 0, eval_strategy = SeaPearl.DFSearch(), single_dive = false, training_timeout = 40000)