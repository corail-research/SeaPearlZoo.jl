include("../common/experiment.jl")
include("../common/utils.jl")
include("comparison.jl")

using TensorBoardLogger, Logging
import CUDA
###############################################################################
######### DFS / RBS comparison on Max Cut
#########  
######### 
###############################################################################

function experiment_Max_Cut_dfs_dive(n, k, n_episodes, n_instances; chosen_features=nothing, feature_size=nothing, n_eval=20, n_eva = n, k_eva = k,n_layers_graph=3, c=2.0, trajectory_capacity=10000, pool = SeaPearl.meanPooling(), nbRandomHeuristics = 1, eval_timeout = 60, restartPerInstances = 1, seedEval = nothing, training_timeout = 3600, eval_every = 120)


SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}
n_step_per_episode = n+1
decay_step = n_step_per_episode*n_episodes*0.6
trajectory_capacity = 800*n_step_per_episode
update_horizon = Int(floor(n_step_per_episode//2))

if isnothing(chosen_features)
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
    feature_size = [6, 5, 2] 
end
rngExp = MersenneTwister(seedEval)
init = Flux.glorot_uniform(MersenneTwister(seedEval))


    agent_dfs= get_heterogeneous_agent(;
    get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=2),        
    get_explorer = () -> get_epsilon_greedy_explorer(decay_step, 0.1; rng = rngExp ),
    batch_size=8,
    update_horizon=update_horizon,
    min_replay_history=Int(round(16*n_step_per_episode//2)),
    update_freq=2,
    target_update_freq=7*n_step_per_episode,
    get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
        feature_size=feature_size,
        conv_size=8,
        dense_size=16,
        output_size=1,
        n_layers_graph=3,
        n_layers_node=2,
        n_layers_output=2, 
        pool=pool,
        σ=NNlib.leakyrelu,
        init = init, 
        #device =gpu
    ),
    γ = 0.99f0
    )
    
    agent_rbs = deepcopy(agent_dfs)

    learnedHeuristics_dfs = OrderedDict{String,SeaPearl.LearnedHeuristic}()
    learnedHeuristics_dfs["dfs"] = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,SeaPearl.DefaultReward,SeaPearl.FixedOutput}(agent_dfs; chosen_features=chosen_features) #Default Reward
    learnedHeuristics_rbs = OrderedDict{String,SeaPearl.LearnedHeuristic}()
    learnedHeuristics_rbs["rbs"] = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,SeaPearl.ScoreReward,SeaPearl.FixedOutput}(agent_rbs; chosen_features=chosen_features) #Score Reward
   
    threshold = 2*k
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()
    MISHeuristic(x; cpmodel=nothing) = length(x.onDomainChange) - 1 < threshold ? 1 : 0
    heuristic_mis = SeaPearl.BasicHeuristic(MISHeuristic)
    selectMax(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.maximum(x.domain)
    heuristic_max = SeaPearl.BasicHeuristic(selectMax)
    basicHeuristics = OrderedDict(
        "MaxCutheuristic" => heuristic_mis,
    )

    generator = SeaPearl.MaxCutGenerator(n,k)
    eval_generator = SeaPearl.MaxCutGenerator(n_eva, k_eva)

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=Int(floor(n_episodes / n_eval)),
        nbInstances=n_instances,
        restartPerInstances=restartPerInstances,
        strategy = SeaPearl.DFSearch(),
        eval_strategy = SeaPearl.DFSearch(),
        out_solver = false,
        generator=generator,
        variableHeuristic=variableHeuristic,
        learnedHeuristics=learnedHeuristics_dfs,
        basicHeuristics=basicHeuristics;
        verbose=true,
        nbRandomHeuristics=nbRandomHeuristics,
        exp_name= "Max_Cut_dfs_"*string(n_episodes)*"_"*string(n)*"_"*string(k)*"_timeout_"*string(training_timeout)*"_eval_every_"*string(eval_every)*"_", 
        eval_generator = eval_generator, 
        seedEval = seedEval,        
        training_timeout = training_timeout,
        eval_every = eval_every,
    )

    println()

    generator = SeaPearl.MaxCutGenerator(n,k)
    eval_generator = SeaPearl.MaxCutGenerator(n_eva, k_eva)
    
    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=Int(floor(n_episodes / n_eval)),
        nbInstances=n_instances,
        restartPerInstances=restartPerInstances,
        strategy = SeaPearl.DFSearch(),
        eval_strategy = SeaPearl.DFSearch(),
        out_solver = true,
        generator=generator,
        variableHeuristic=variableHeuristic,
        learnedHeuristics=learnedHeuristics_rbs,
        basicHeuristics=basicHeuristics;
        verbose=true,
        nbRandomHeuristics=nbRandomHeuristics,
        exp_name= "Max_Cut_rbs_"*string(n_episodes)*"_"*string(n)*"_"*string(k)*"_timeout_"*string(training_timeout)*"_eval_every_"*string(eval_every)*"_",
        eval_timeout=eval_timeout, 
        eval_generator = eval_generator, 
        seedEval = seedEval,
        training_timeout = training_timeout,
        eval_every = eval_every,
    )
    nothing

end
###############################################################################
######### Simple MaxCut experiment
#########  
######### 
###############################################################################

function simple_experiment_MaxCut(n, k, n_episodes, n_instances; chosen_features=nothing, feature_size=nothing, n_eval=20, n_eva = n, k_eva = k,n_layers_graph=3, reward = SeaPearl.GeneralReward, c=2.0, trajectory_capacity=30000, pool = SeaPearl.meanPooling(), nbRandomHeuristics = 1, eval_timeout = 240, restartPerInstances = 1, seedEval = nothing, device=cpu, batch_size = 64, update_freq = 10,  target_update_freq= 500, name = "", numDevice = 0, eval_strategy = SeaPearl.DFSearch())

    #to change of device : CUDA.device!(i) i is the id tof the GPU being used

    n_step_per_episode = Int(round(n/5))
    update_horizon = Int(round(n_step_per_episode//2))

    if device == gpu
        CUDA.device!(numDevice)
    end

    generator = SeaPearl.MaxCutGenerator(n,k)
    eval_generator = SeaPearl.MaxCutGenerator(n_eva, k_eva)
    
    evalFreq=Int(floor(n_episodes / n_eval))

    step_explorer = Int(floor(n_episodes*n_step_per_episode*0.1 ))

    if isnothing(chosen_features)
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
        feature_size = [6, 5, 2] 
    end

    rngExp = MersenneTwister(seedEval)
    init = Flux.glorot_uniform(MersenneTwister(seedEval))

    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    selectMax(x::SeaPearl.BoolVar; cpmodel=nothing) = SeaPearl.maximum(x.domain.inner)
    selectMin(x::SeaPearl.BoolVar; cpmodel=nothing) = SeaPearl.minimum(x.domain.inner)
    heuristic_max = SeaPearl.BasicHeuristic(selectMax)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "expert_max" => heuristic_max,
        "expert_min" => heuristic_min
    )

    agent_3 = get_heterogeneous_agent(;
    get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=2),        
    get_explorer = () -> get_epsilon_greedy_explorer(step_explorer, 0.01; rng = rngExp ),
    batch_size=batch_size,
    update_horizon=update_horizon,
    min_replay_history=Int(round(16*n_step_per_episode//2)),
    update_freq=update_freq,
    target_update_freq=target_update_freq,
    get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
        feature_size=feature_size,
        conv_size=8,
        dense_size=16,
        output_size=1,
        n_layers_graph=3,
        n_layers_node=3,
        n_layers_output=2, 
        pool=pool,
        σ=NNlib.leakyrelu,
        init = init,
        device = device
    ),
    γ =  0.99f0
    )

    learned_heuristic_3 = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_3; chosen_features=chosen_features)

    learnedHeuristics = OrderedDict(
        "3layer" => learned_heuristic_3,
    )

    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=evalFreq,
        nbInstances=n_instances,
        restartPerInstances=restartPerInstances,
        generator=generator,
        eval_strategy=eval_strategy,
        variableHeuristic=variableHeuristic,
        learnedHeuristics=learnedHeuristics,
        basicHeuristics=basicHeuristics;
        out_solver=true,
        verbose=true,
        seedEval=seedEval,
        nbRandomHeuristics=nbRandomHeuristics,
        exp_name=name *'_'* string(n) *"_"*string(n_eva) * "_" * string(n_episodes) * "_"* string(seedEval) * "_",
        eval_timeout=eval_timeout, 
        eval_generator=eval_generator,
        device = device
    )
    nothing

end


###############################################################################
######### Max cut CPNN vs FFCPNN
#########  
######### 
###############################################################################

function simple_MaxCut_cpnn(n, k, n_episodes, n_instances; chosen_features=nothing, feature_size=nothing, n_eval=20, n_eva = n, k_eva = k,n_layers_graph=3, reward = SeaPearl.GeneralReward, c=2.0, trajectory_capacity=5000, pool = SeaPearl.meanPooling(), nbRandomHeuristics = 1, eval_timeout = 60, restartPerInstances = 10, seedEval = nothing, eval_strategy = SeaPearl.ILDSearch(2))
    """
    Runs a single experiment on MIS
    """
    n_step_per_episode = n+1
    reward = SeaPearl.GeneralReward
    generator = SeaPearl.MaxCutGenerator(n,k)
    eval_generator = SeaPearl.MaxCutGenerator(n_eva, k_eva)
    
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    trajectory_capacity = 800*n_step_per_episode
    update_horizon = Int(round(n_step_per_episode//2))
    learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()

    if isnothing(chosen_features)
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
        feature_size = [6, 5, 2] 
    end
    rngExp = MersenneTwister(seedEval)
    init = Flux.glorot_uniform(MersenneTwister(seedEval))

    agent_ffcpnn= get_heterogeneous_agent(;
    get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=2),        
    get_explorer = () -> get_epsilon_greedy_explorer(Int(floor(n_episodes*n_step_per_episode*0.80)), 0.1; rng = rngExp ),
    batch_size=32,
    update_horizon=update_horizon,
    min_replay_history=Int(round(16*n_step_per_episode//2)),
    update_freq=4,
    target_update_freq=7*n_step_per_episode,
    get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
        feature_size=feature_size,
        conv_size=16,
        dense_size=16,
        output_size=1,
        n_layers_graph=n_layers_graph,
        n_layers_node=3,
        n_layers_output=3, 
        pool=SeaPearl.meanPooling(),
        σ=NNlib.leakyrelu,
        init = init, 
        #device =gpu
    ),
    γ = 0.99f0
    )

    agent_cpnn= get_heterogeneous_agent(;
    get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=2),        
    get_explorer = () -> get_epsilon_greedy_explorer(Int(floor(n_episodes*n_step_per_episode*0.80)), 0.1; rng = rngExp ),
    batch_size=32,
    update_horizon=update_horizon,
    min_replay_history=Int(round(16*n_step_per_episode//2)),
    update_freq=4,
    target_update_freq=7*n_step_per_episode,
    get_heterogeneous_nn = () -> get_heterogeneous_cpnn(
        feature_size=feature_size,
        conv_size=16,
        dense_size=16,
        output_size=2,
        n_layers_graph=n_layers_graph,
        n_layers_output=3,
        pool=SeaPearl.meanPooling(),
        σ=NNlib.leakyrelu,
        init = init, 
        #device =gpu
    ),
    γ = 0.99f0
    )

    learned_heuristic_ffcpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_ffcpnn, chosen_features=chosen_features)
    learned_heuristic_cpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_cpnn; chosen_features=chosen_features)
     
    learnedHeuristics["ffcpnn"] = learned_heuristic_ffcpnn
    learnedHeuristics["cpnn"] = learned_heuristic_cpnn
    
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()
    selectMax(x::SeaPearl.BoolVar; cpmodel=nothing) = SeaPearl.maximum(x.domain.inner)
    selectMin(x::SeaPearl.BoolVar; cpmodel=nothing) = SeaPearl.minimum(x.domain.inner)
    heuristic_max = SeaPearl.BasicHeuristic(selectMax)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "expert_max" => heuristic_max,
        "expert_min" => heuristic_min
    )

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=Int(floor(n_episodes / n_eval)),
        nbInstances=n_instances,
        restartPerInstances=restartPerInstances,
        eval_strategy = eval_strategy,
        #strategy = SeaPearl.DFWBSearch(),
        #out_solver = false,
        generator=generator,
        variableHeuristic=variableHeuristic,
        learnedHeuristics=learnedHeuristics,
        basicHeuristics=basicHeuristics;
        verbose=true,
        nbRandomHeuristics=nbRandomHeuristics,
        exp_name= "MaxCut_ffcpnn_cpnn_"*string(n_episodes)*"_"*string(n)*"_"*string(k)*"->"*string(n_eva)*"_"*string(k_eva)*"_"* string(n_episodes),
        eval_timeout=eval_timeout, 
        eval_generator = eval_generator, 
        seedEval = seedEval
    )
    nothing

end