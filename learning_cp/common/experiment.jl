using SeaPearl
using SeaPearlExtras
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux
using JSON
using BSON: @save, @load
using Dates
using Random
using LightGraphs
using OrderedCollections
using CircularArrayBuffers
using PyCall
using Pkg
using TensorBoardLogger, Logging

"""
ENV["PYTHON"] = "/home/x86_64-unknown-linux_ol7-gnu/anaconda-2022.05/bin/python"
using Pkg
Pkg.build("PyCall")
run(`$(PyCall.python) -m pip install matplotlib`)
run(`$(PyCall.python) -m pip install pandas`)
run(`$(PyCall.python) -m pip install seaborn`)
run(`$(PyCall.python) -m pip install ipython`)
"""

# -------------------
# -------------------
# Core function
# -------------------
# -------------------

function generate_graph_txt(graph)
    str =""
    str *=string(length(graph.fadjlist))*' '*string(graph.ne)*"\n"
    for neighbors in graph.fadjlist
        str *= join(neighbors," ")*"\n"
    end
    return str
end

function trytrain(; nbEpisodes::Int, evalFreq::Int, nbInstances::Int, restartPerInstances::Int=1, generator::SeaPearl.AbstractModelGenerator, variableHeuristic::SeaPearl.AbstractVariableSelection=SeaPearl.MinDomainVariableSelection{false}(), learnedHeuristics::OrderedDict{String,<:SeaPearl.LearnedHeuristic}, basicHeuristics::OrderedDict{String,SeaPearl.BasicHeuristic}, base_name="experiment"::String, exp_name=""::String, out_solver=true::Bool, verbose=false::Bool, nbRandomHeuristics=0::Int, eval_timeout=nothing::Union{Nothing, Int},  training_timeout=nothing::Union{Nothing, Int}, eval_every =nothing::Union{Nothing, Int}, eval_strategy=SeaPearl.DFSearch(), strategy = SeaPearl.DFSearch(), seedTraining = nothing::Union{Nothing, Int}, seedEval =  nothing, eval_generator=nothing, logger = nothing, device = cpu)


    experienceTime = Base.replace("$(round(now(), Dates.Second(3)))", ":" => "-")
    date = split(experienceTime, "T")[1]
    time = split(experienceTime, "T")[2]
    logger =TBLogger("tensorboard_logs/"*exp_name*date*"_"*time, min_level=Logging.Info)

    if !isdir(date)
        mkdir(date)
    end
    dir = mkdir(string(date, "/exp_", exp_name, time))
    lh = last(collect(values(learnedHeuristics)))
    code_dir = mkdir(dir*"/code/")
    for file in readdir(".")
        if isfile(file)
            cp(file, code_dir*file)
        end
    end

    randomHeuristics = Array{SeaPearl.BasicHeuristic}(undef, 0)
    for i in 1:nbRandomHeuristics
        push!(randomHeuristics, SeaPearl.RandomHeuristic())
    end

    valueSelectionArray = cat(collect(values(learnedHeuristics)), collect(values(basicHeuristics)), randomHeuristics, dims=1)

    if !isnothing(eval_generator)
        evaluator = SeaPearl.SameInstancesEvaluator(valueSelectionArray, eval_generator; evalFreq=evalFreq, nbInstances=nbInstances, evalTimeOut = eval_timeout, rng = MersenneTwister(seedEval) )
    else
        evaluator = SeaPearl.SameInstancesEvaluator(valueSelectionArray, generator; evalFreq=evalFreq, nbInstances=nbInstances, evalTimeOut = eval_timeout, rng = MersenneTwister(seedEval))
    end

    eval_instances_dir = mkdir(dir*"/eval_instances/")
    for (idx,instance) in enumerate(evaluator.instances)

        if !isnothing(instance.adhocInfo)
            open(eval_instances_dir*"instance_"*string(idx)*".txt", "w") do io
                write(io, generate_graph_txt(instance.adhocInfo))
            end
        end  
    end

    metricsArray, eval_metricsArray = SeaPearl.train!(
        valueSelectionArray=valueSelectionArray,
        generator=generator,
        nbEpisodes=nbEpisodes,
        strategy=strategy,
        eval_strategy = eval_strategy,
        variableHeuristic=variableHeuristic,
        out_solver=out_solver,
        verbose=verbose,
        evaluator = evaluator,
        training_timeout = training_timeout,
        restartPerInstances=restartPerInstances,
        rngTraining = MersenneTwister(seedTraining), 
        eval_every = eval_every, 
        logger = logger,
        device = device,
    )

    #saving model weights
    for (key, lh) in learnedHeuristics
        if (hasfield(typeof(lh.agent.policy),:approximator)) #PPO
            model = Flux.cpu(lh.agent.policy.approximator)
        else #DQN
            model = Flux.cpu(lh.agent.policy.learner.approximator)
        end
        @save dir * "/model_" * key * ".bson" model
    end

    counter = 0
    for key in keys(learnedHeuristics)
        counter += 1
        SeaPearlExtras.storedata(metricsArray[counter]; filename=dir * "/" * base_name * "_training_" * key)
    end

    counter = 0
    for key in keys(learnedHeuristics)
        counter += 1
        SeaPearlExtras.storedata(eval_metricsArray[:, counter]; filename=dir * "/" * base_name * "_" * key)
    end
    for key in keys(basicHeuristics)
        counter += 1
        SeaPearlExtras.storedata(eval_metricsArray[:, counter]; filename=dir * "/" * base_name * "_" * key)
    end
    for i = 1:nbRandomHeuristics
        SeaPearlExtras.storedata(eval_metricsArray[:, counter+i]; filename=dir * "/" * base_name * "_random$(i)")
    end

    py"""
    import sys
    sys.path.insert(0, "/home/martom/SeaPearl/SeaPearlExtras.jl/src/metrics/basicmetrics/")
    
    from benchmarkPy import *
    import plots
    import numpy as np
    def benchmark(path):
        print_all(path +"/")

    def plot(path,eval):
        plots.all(path +"/", window=100, estimator=np.mean, ilds= eval)
    """

    py"plot"(dir, eval_strategy != SeaPearl.DFSearch())

    chosen_features = valueSelectionArray[1].chosen_features
    
    folder = "/home/martom/SeaPearl/SeaPearlZoo/learning_cp/comparison/experiment_for_paper/Final_Benchmark/GC/S/exp_009_maxwell06_final_benchmark_GC_S_20_20_5000_165_00-53-39_better/"

    chosen_features = Dict(
        "node_number_of_neighbors" => true,
        "constraint_type" => true,
        "constraint_activity" => true,
        "nb_not_bounded_variable" => true,
        "variable_initial_domain_size" => true,
        "variable_domain_size" => true,
        "variable_is_objective" => true,
        "variable_assigned_value" => true,
        "variable_is_bound" => true,
        "values_raw" => true)
    
    generator = SeaPearl.ClusterizedGraphColoringGenerator(20,5,0.9)
    n = 20 # Number of instances to evaluate on
    budget = 10000 # Budget of visited nodes
    has_objective = false # Set it to true if we have to branch on the objective variable
    eval_strategy = SeaPearl.ILDSearch(2)
    include_dfs = (eval_strategy == SeaPearl.DFSearch()) # Set it to true if you want to evaluate with DFS in addition to ILDS
    
    basicHeuristics = OrderedDict(
        "random" => SeaPearl.RandomHeuristic()
        )

    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "random" => SeaPearl.RandomHeuristic()
        )
    include("../common/benchmark.jl")
    include("../common/performanceprofile.jl")
    Base.invokelatest(benchmark, dir, n, chosen_features, has_objective, generator, basicHeuristics, include_dfs, budget; ILDS = eval_strategy)
    #Base.invokelatest(performanceProfile, dir, n, chosen_features, has_objective, generator, basicHeuristics)

    py"""
    import sys
    sys.path.insert(0, "/home/martom/SeaPearl/SeaPearlExtras.jl/src/metrics/basicmetrics/")
    
    from benchmarkPy import *
    import plots
    import numpy as np
    def benchmark(path):
        print_all(path +"/")

    def plot(path,eval):
        plots.all(path +"/", window=100, estimator=np.mean, ilds= eval)
    def performance(path):
        plots.performance(path +"/", ilds= eval)
    """

    print(dir)
    py"benchmark"(dir)
    py"performance"(dir)

    return metricsArray, eval_metricsArray
end