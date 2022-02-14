using Test
include("../../classic_cp/kidneyexchange/kidneyexchange.jl")

const pathToData = "../classic_cp/kidneyexchange/data/"
instancesToSolve = ["kep_13_0.1", "kep_20_0.3", "kep_40_0.05"]
expectedBestScores = [6, 20, 29]

@testset "kidneyexchange.jl" begin
    @testset "reduce_instance()" begin
        # reductibleInstance
        reductibleInstance = getInputData(pathToData*"kep_13_0.1")
        canReduce, pairsEquivalence, reduced_c = reduce_instance(reductibleInstance)
        expectedPairsEquivalence = [2, 3, 4, 11, 12, 13]
        expectedReduced_c = [[3], [1], [2], [6], [4], [5]]
        @test canReduce
        @test expectedPairsEquivalence == pairsEquivalence
        @test expectedReduced_c == reduced_c

        # unreductibleInstance
        unreductibleInstance = getInputData(pathToData*"kep_8_0.2")
        canReduce, pairsEquivalence, reduced_c = reduce_instance(unreductibleInstance)
        @test !canReduce
    end

    @testset "solve_kidneyexchange_matrix()" begin
        for i in 1:length(instancesToSolve)
            solved_model = solve_kidneyexchange_matrix(pathToData*instancesToSolve[i])
            solutions = solved_model.statistics.solutions
            numberOfPairs = trunc(Int, sqrt(length(solved_model.variables) - 1))
            realSolutions = filter(e -> !isnothing(e),solutions)
            bestScores = map(e -> -minimum(values(e)),realSolutions)
            bestScore = maximum(bestScores)
            @test bestScore == expectedBestScores[i]
        end
    end

    @testset "solve_kidneyexchange_vector()" begin
        for i in 1:length(instancesToSolve)
            solved_model = solve_kidneyexchange_vector(pathToData*instancesToSolve[i])
            solutions = solved_model.statistics.solutions
            numberOfPairs = trunc(Int, sqrt(length(solved_model.variables) - 1))
            realSolutions = filter(e -> !isnothing(e),solutions)
            bestScores = map(e -> -minimum(values(e)),realSolutions)
            bestScore = maximum(bestScores)
            @test bestScore == expectedBestScores[i]
        end
    end
end