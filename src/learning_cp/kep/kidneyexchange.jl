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

struct select_random_value <: Function
    rng::MersenneTwister
    function select_random_value(rng::MersenneTwister)
        return new(rng)
    end
end
function (func::select_random_value)(x::SeaPearl.IntVar; cpmodel=nothing)

    selected_number = rand(func.rng, 1: length(x.domain))
    i = 1
    for value in x.domain
        if i == selected_number
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
    if !isnothing(args.seed_eval)
        rngEval = MersenneTwister(args.seed_eval)
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
        nbEpisodes=args.num_episodes,
        strategy=args.strategy,
        eval_strategy=args.eval_strategy,
        variableHeuristic=variableSelection,
        out_solver = true,
        verbose = true,
        evaluator=SeaPearl.SameInstancesEvaluator(heuristics, evalGenerator;
            evalFreq = div(args.num_episodes, args.num_evals), 
            nbInstances = args.num_instances, 
            evalTimeOut = args.eval_timeout
        ),
        metrics = nothing, 
        restartPerInstances = args.num_restarts_per_instance,
    )
    println("\n\nExperiment done, writing weights")
    trainedWeights = Flux.params(agent.policy.learner.approximator.model)
    @save dir*"/model_weights_kep"*string(args.num_nodes)*".bson" trainedWeights
    println("Writing files")

    SeaPearlExtras.storedata(metricsArray[1]; filename=dir*"/kep_$(args.num_nodes)_training")
    SeaPearlExtras.storedata(evalMetricsArray[:,1]; filename=dir*"/kep_$(args.num_nodes)_trained")
    for i = 1:args.num_random_heuristics
        SeaPearlExtras.storedata(evalMetricsArray[:,i+2]; filename=dir*"/kep_$(args.num_nodes)_random$(i)")
    end

    return metricsArray, evalMetricsArray
end

function main(args::KepParameters)


    # -------------------
    # Generator
    # ------------------- 
    kepGenerator = SeaPearl.KepGenerator(args.num_nodes, args.density)
    kepEvalGenerator = SeaPearl.KepGenerator(args.num_nodes, args.density)

    SR = SeaPearl.DefaultStateRepresentation{KepFeaturization, SeaPearl.DefaultTrajectoryState}
    args.SR = SR
    args.num_input_features = SeaPearl.feature_length(SR)

    # -------------------
    # Agent definition
    # -------------------   
    agent = create_agent(args)

    # -------------------
    # Value Heuristic definition
    # -------------------
    learnedHeuristic = SeaPearl.SimpleLearnedHeuristic{SR, args.reward, SeaPearl.FixedOutput}(agent)
    basic_heuristic = SeaPearl.BasicHeuristic()

    rngHeuristic = MersenneTwister(33)
    randomHeuristics = []
    for i in 1:args.num_random_heuristics
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
        "--num_nodes"
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
        "--seed_eval"
            help = "seed used to generate eval instances"
            arg_type = Int
            default = 15

    end
    return parse_args(s; as_symbols=true)
end

function script()
    args = parse_commandline()
    kepParams = KepParameters()
    kepParams.num_episodes = args[:n]
    kepParams.num_restarts_per_instance = args[:r]
    kepParams.eval_timeout = args[:timeout]
    kepParams.nbInstances = args[:i]
    kepParams.num_random_heuristics = args[:random]
    kepParams.num_nodes = args[:num_nodes]
    kepParams.num_nodesEval = args[:num_nodes]
    kepParams.seed = args[:seed]
    kepParams.seed_eval = args[:seed_eval]
    kepParams.density = args[:density]
    kepParams.reward = SeaPearl.GeneralReward

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
        args.num_episodes = round(total_ep/r)
        args.num_restarts_per_instance = round(r)
        main(args)
    end
end

script()

nothing