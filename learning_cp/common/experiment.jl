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
using ArgParse
using CircularArrayBuffers

# -------------------
# -------------------
# Core function
# -------------------
# -------------------

function trytrain(; nbEpisodes::Int, evalFreq::Int, nbInstances::Int, restartPerInstances::Int=1, generator::SeaPearl.AbstractModelGenerator, variableHeuristic::SeaPearl.AbstractVariableSelection=SeaPearl.MinDomainVariableSelection{false}(), learnedHeuristics::OrderedDict{String,<:SeaPearl.LearnedHeuristic}, basicHeuristics::OrderedDict{String,SeaPearl.BasicHeuristic}, expParameters=Dict{String,Any}()::Dict{String,Any}, base_name="experiment"::String, exp_name=""::String, out_solver=true::Bool, verbose=false::Bool, nbRandomHeuristics=0::Int, eval_timeout=nothing::Union{Nothing, Int})
    experienceTime = now()
    dir = mkdir(string("exp_", exp_name, Base.replace("$(round(experienceTime, Dates.Second(3)))", ":" => "-")))
    lh = last(collect(values(learnedHeuristics)))
    if isa(lh.agent.policy.explorer,RL.EpsilonGreedyExplorer)
        explorer_params =  :explorerParameters => Dict(
            :ϵ_stable => lh.agent.policy.explorer.ϵ_stable,
            :decay_steps => lh.agent.policy.explorer.decay_steps,
        )
    elseif isa(lh.agent.policy.explorer,RL.UCBExplorer)
        explorer_params =  :explorerParameters => Dict(
            :actioncounts => lh.agent.policy.explorer.actioncounts,
            :c => lh.agent.policy.explorer.c,
        )
    end
    commonExpParameters = Dict(
        :experimentParameters => Dict(
            :nbEpisodes => nbEpisodes,
            :restartPerInstances => restartPerInstances,
            :evalFreq => evalFreq,
            :nbInstances => nbInstances,
        ),
        :nbRandomHeuristics => nbRandomHeuristics,
        :Featurization => Dict(
            :featurizationType => typeof(lh).parameters[1].parameters[1],
            :chosen_features => lh.chosen_features
        ),
        :learnerParameters => Dict(
            :model => string(lh.agent.policy.learner.approximator.model),
            :gamma => lh.agent.policy.learner.sampler.γ,
            :batch_size => lh.agent.policy.learner.sampler.batch_size,
            :update_horizon => lh.agent.policy.learner.sampler.n,
            :min_replay_history => lh.agent.policy.learner.min_replay_history,
            :update_freq => lh.agent.policy.learner.update_freq,
            :target_update_freq => lh.agent.policy.learner.target_update_freq,
        ),
        explorer_params,
        :trajectoryParameters => Dict(
            :trajectoryType => typeof(lh.agent.trajectory),
            :capacity => CircularArrayBuffers.capacity(lh.agent.trajectory.traces.action) - 1
        ),
        :reward => typeof(lh).parameters[2]
    )

    expParameters = merge(expParameters, commonExpParameters)
    open(dir * "/params.json", "w") do file
        JSON.print(file, expParameters)
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
        variableHeuristic=variableHeuristic,
        out_solver=out_solver,
        verbose=verbose,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray, generator; evalFreq=evalFreq, nbInstances=nbInstances, evalTimeOut = eval_timeout),
        restartPerInstances=restartPerInstances
    )

    #saving model weights
    for (key, lh) in learnedHeuristics
        model = lh.agent.policy.learner.approximator
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
