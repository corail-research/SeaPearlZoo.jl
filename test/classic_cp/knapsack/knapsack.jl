@testset "knapsack.jl" begin
    SeaPearlZoo.build_model_and_solve_knapsack(SeaPearlZoo.parseKnapsackFile!("classic_cp/knapsack/ks_4_0"))
end