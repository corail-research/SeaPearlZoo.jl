@testset "learning_kep.jl" begin
    kep_parameters = SeaPearlZoo.KepParameters()
    kep_parameters.num_episodes = 1
    kep_parameters.density = 1
    kep_parameters.num_nodes = 2
    kep_parameters.num_nodes_eval = 2
    kep_parameters.num_restarts_per_instance = 1
    kep_parameters.num_evals = 1
    kep_parameters.num_instances = 1
    SeaPearlZoo.main_learning_kep(kep_parameters)
end