using Test
using SeaPearl
include("../../classic_cp/kidneyexchange/kidneyexchange.jl")

# TODO why error with getInputData("data/someInstace")?
reductibleInstance = """13 0.1

1 4 5
2
3 9
6 8
7


10

13
11
12"""

reductibleInstance = parseInput(reductibleInstance)

unreductibleInstance = """8 0.2
2 3
1 6
1 4 7
2
2
5
8
3"""

unreductibleInstance = parseInput(unreductibleInstance)

@testset "kidneyexchange.jl" begin
    @testset "reduce_instance()" begin
        
        # reductibleInstance
        canReduce, pairsEquivalence, reduced_c = reduce_instance(reductibleInstance)
        
        @test canReduce
        @test length(pairsEquivalence) < reductibleInstance.numberOfPairs
        @test length(pairsEquivalence) == length(reduced_c)

        # unreductibleInstance
        canReduce, pairsEquivalence, reduced_c = reduce_instance(unreductibleInstance)

        @test !canReduce
    end
end