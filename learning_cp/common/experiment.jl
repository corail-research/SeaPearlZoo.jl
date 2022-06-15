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

# -------------------
# -------------------
# Core function
# -------------------
# -------------------

function trytrain(; nbEpisodes::Int, evalFreq::Int, nbInstances::Int, restartPerInstances::Int=1, generator::SeaPearl.AbstractModelGenerator, variableHeuristic::SeaPearl.AbstractVariableSelection=SeaPearl.MinDomainVariableSelection{false}(), learnedHeuristics::OrderedDict{String,<:SeaPearl.LearnedHeuristic}, basicHeuristics::OrderedDict{String,SeaPearl.BasicHeuristic}, base_name="experiment"::String, exp_name=""::String, out_solver=true::Bool, verbose=false::Bool, nbRandomHeuristics=0::Int, eval_timeout=nothing::Union{Nothing, Int}, eval_strategy=SeaPearl.DFSearch())
    experienceTime = now()
    dir = mkdir(string("exp_", exp_name, Base.replace("$(round(experienceTime, Dates.Second(3)))", ":" => "-")))
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

    metricsArray, eval_metricsArray = SeaPearl.train!(
        valueSelectionArray=valueSelectionArray,
        generator=generator,
        nbEpisodes=nbEpisodes,
        strategy=SeaPearl.DFSearch(),
        eval_strategy = eval_strategy,
        variableHeuristic=variableHeuristic,
        out_solver=out_solver,
        verbose=verbose,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray, generator; evalFreq=evalFreq, nbInstances=nbInstances, evalTimeOut = eval_timeout),
        restartPerInstances=restartPerInstances
    )

    #saving model weights
    for (key, lh) in learnedHeuristics
        if (hasfield(typeof(lh.agent.policy),:approximator)) #PPO
            model = lh.agent.policy.approximator
        else #DQN
            model = lh.agent.policy.learner.approximator
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

    return metricsArray, eval_metricsArray
end
