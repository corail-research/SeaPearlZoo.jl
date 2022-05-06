import Pkg
Pkg.activate("")
Pkg.instantiate()

using ArgParse
using SeaPearl
using SeaPearlExtras
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using Statistics
using Random
using BSON: @load, @save
using Dates
using JSON
using GeometricFlux

include("agents.jl")
include("features.jl")

mutable struct KepParameters
    nbNodes             ::Int
    nbNodesEval         ::Int
    density             ::Float64
    nbEpisodes          ::Int 
    restartPerInstance  ::Int
    nbEvals             ::Int
    nbInstances         ::Int
    nbRandomHeuristics  ::Int
    evalTimeout         ::Int
    batchSize           ::Int
    seed                ::Union{Nothing,Int}
    seedEval            ::Union{Nothing,Int}
    reward              ::Type{<:SeaPearl.AbstractReward}
    strategy            ::SeaPearl.SearchStrategy
    evalStrategy        ::SeaPearl.SearchStrategy
    SR                  ::Union{Nothing,Type{<:SeaPearl.AbstractStateRepresentation}}
    numInFeatures       ::Union{Nothing,Int}

    # Experiment default values
    function KepParameters()
        return new(
            10,         #nbNodes
            10,         #nbNodeEval
            0.2,        #density
            500,       #nbEpisodes #usually 1000
            20,         #restartPerInstance
            20,         #nbEval
            10,         #nbInstances
            1,          #nbRandomHeuristics #usually 5
            300,        #evalTimeOut
            1,          #batchSize
            nothing,       #seed
            nothing,       #seedEval
            SeaPearl.ExperimentalReward,
            SeaPearl.DFSearch(),
            SeaPearl.DFSearch(),
            nothing,
            nothing
        )
    end
end

struct select_random_value <: Function
    rng::MersenneTwister
    function select_random_value(rng::MersenneTwister)
        return new(rng)
    end
end
function (func::select_random_value)(x::SeaPearl.IntVar; cpmodel=nothing)

    #print("selected variable : ", x)
    selected_number = rand(func.rng, 1: length(x.domain))
    i = 1
    for value in x.domain
        if i == selected_number
            #println(" : selected value : ",value)
            return value
        end
        i += 1
    end
    @assert false "This should not happen"
end

function trytrain(
        args::KepParameters,
        generator::SeaPearl.KepGenerator,      
        evalGenerator::SeaPearl.KepGenerator,    
        heuristics::Vector{<:SeaPearl.ValueSelection},
        variableSelection::SeaPearl.AbstractVariableSelection,
        agent::RL.Agent
    )
    println("Creating files")

    experienceTime = now()
    dir = mkdir(string("exp_",Base.replace("$(round(experienceTime, Dates.Second(3)))",":"=>"-")))

    open(dir*"/params.json", "w") do file
        JSON.print(file, Dict(fn=>string(getfield(args, fn)) for fn ∈ fieldnames(KepParameters)))
    end

    println("Setting seed for evaluation")
    if !isnothing(args.seedEval)
        rngEval = MersenneTwister(args.seedEval)
    else
        rngEval = MersenneTwister()
    end
    println("Setting seed for training")
    if !isnothing(args.seed)
        rngTraining = MersenneTwister(args.seed)
    else
        rngTraining = MersenneTwister()
    end   

    println("Training start")
    metricsArray, evalMetricsArray = SeaPearl.train!(
        valueSelectionArray=heuristics,
        generator=generator,
        nbEpisodes=args.nbEpisodes,
        strategy=args.strategy,
        eval_strategy=args.evalStrategy,
        variableHeuristic=variableSelection,
        out_solver = true,
        verbose = true,
        evaluator=SeaPearl.SameInstancesEvaluator(heuristics, evalGenerator;
            seed = args.seedEval,
            evalFreq = div(args.nbEpisodes,args.nbEvals), 
            nbInstances = args.nbInstances, 
            evalTimeOut = args.evalTimeout
        ),
        metrics = nothing, 
        restartPerInstances = args.restartPerInstance,
    )
    println("\n\nExperiment done, writing weights")
    trainedWeights = Flux.params(agent.policy.learner.approximator.model)
    @save dir*"/model_weights_kep"*string(args.nbNodes)*".bson" trainedWeights
    println("Writing files")

    SeaPearlExtras.storedata(metricsArray[1]; filename=dir*"/kep_$(args.nbNodes)_training")
    SeaPearlExtras.storedata(evalMetricsArray[:,1]; filename=dir*"/kep_$(args.nbNodes)_trained")
    for i = 1:args.nbRandomHeuristics
        SeaPearlExtras.storedata(evalMetricsArray[:,i+2]; filename=dir*"/kep_$(args.nbNodes)_random$(i)")
    end

    return metricsArray, evalMetricsArray
end


function main(args::KepParameters)

    

    # -------------------
    # Generator
    # ------------------- 
    kepGenerator = SeaPearl.KepGenerator(args.nbNodes, args.density)
    kepEvalGenerator = SeaPearl.KepGenerator(args.nbNodes, args.density)

    SR = SeaPearl.DefaultStateRepresentation{KepFeaturization, SeaPearl.DefaultTrajectoryState}
    args.SR = SR
    args.numInFeatures = SeaPearl.feature_length(SR)

    

    # -------------------
    # Agent definition
    # -------------------   
    agent = create_agent(args)


    # -------------------
    # Value Heuristic definition
    # -------------------
    learnedHeuristic = SeaPearl.LearnedHeuristic{SR, args.reward, SeaPearl.FixedOutput}(agent)
    basic_heuristic = SeaPearl.BasicHeuristic()

    rngHeuristic = MersenneTwister(33)
    randomHeuristics = []
    for i in 1:args.nbRandomHeuristics
        push!(randomHeuristics, SeaPearl.BasicHeuristic(select_random_value(rngHeuristic)))
    end

    valueSelectionArray = [learnedHeuristic, basic_heuristic]
    append!(valueSelectionArray, randomHeuristics)


    variableSelection = SeaPearl.MinDomainVariableSelection()
    
    # -------------------
    # -------------------
    # Core function
    # -------------------
    # -------------------
    
    trytrain(args, kepGenerator, kepEvalGenerator, valueSelectionArray, variableSelection, agent)
end

#----------------
# Scirpt files
#----------------
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "-n"
            help = "Number of training instances"
            arg_type = Int
            default = 1000
        "-r"
            help = "Number of restarts per instance"
            arg_type = Int
            default = 1
        "--timeout"
            help = "Evalutation timeout"
            arg_type = Int
            default = 300
        "-i"
            help = "Number of evaluation instances"
            arg_type = Int
            default = 3
        "--random"
            help = "Number of random heuristic for evaluation"
            arg_type = Int
            default = 3
        "--nbNodes"
            help = "Number of nodes for the generated instances"
            arg_type = Int
            default = 20
        "--density"
            help = "Density for generated evaluation instances"
            arg_type = Float64
            default = 0.1
        "--reward"
            help = "Name of the reward: CPReward | SmartReward"
            arg_type = String
            default = "ExperimentalReward"
        "--seed"
            help = "seed used to generate model and randomheuristic"
            arg_type = Int
            default = 15
        "--seedEval"
            help = "seed used to generate eval instances"
            arg_type = Int
            default = 15

    end
    return parse_args(s; as_symbols=true)
end

function script()
    args = parse_commandline()
    kepParams = KepParameters()
    kepParams.nbEpisodes = args[:n]
    kepParams.restartPerInstance = args[:r]
    kepParams.evalTimeout = args[:timeout]
    kepParams.nbInstances = args[:i]
    kepParams.nbRandomHeuristics = args[:random]
    kepParams.nbNodes = args[:nbNodes]
    kepParams.nbNodesEval = args[:nbNodes]
    kepParams.seed = args[:seed]
    kepParams.seedEval = args[:seedEval]
    kepParams.density = args[:density]
    kepParams.reward = SeaPearl.ExperimentalReward

    main(kepParams)
end

function scriptDebug(nbEpisodes, restartPerInstance, timeout, nbInstances, random, nbNodes, density, seed, seedEval)
    kepParams = KepParameters()
    kepParams.nbEpisodes = nbEpisodes
    kepParams.restartPerInstance = restartPerInstance
    kepParams.evalTimeout = timeout
    kepParams.nbInstances = nbInstances
    kepParams.nbRandomHeuristics = random
    kepParams.nbNodes = nbNodes
    kepParams.nbNodesEval = nbNodes
    kepParams.seed = seed
    kepParams.seedEval = seedEval
    kepParams.density = density
    kepParams.reward = SeaPearl.ExperimentalReward 

    main(kepParams)
end

# -------------------
# Episode/Restart ratio experiment
# -------------------
function exp_restart()
    restarts = exp10.(range(0,3,length=5))
    for r in restarts
        println("Starting experiment for restarts = $r")
        total_ep = 50000
        args = KepParameters()
        args.nbEpisodes = round(total_ep/r)
        args.restartPerInstance = round(r)
        main(args)
    end
end

script()
# scriptDebug(1000, 1, 100, 50, 1, 10, 0.1, 15, 15) # nbEpisodes, restartPerInstance, timeout, nbInstances, random, nbNodes, density, seed, seedEval
