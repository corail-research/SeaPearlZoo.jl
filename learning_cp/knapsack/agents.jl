fixedGCNargs = CPRL.ArgsVariableOutputGCNLSTM( 
    lastLayer = 20,
    numInFeatures = numberOfFeatures,
    firstHiddenGCN = 20,
    secondHiddenGCN = 20,
    hiddenDense = 20,
    lstmSize = 20
) 


agent = RL.Agent(
    policy = RL.QBasedPolicy(
        learner = CPRL.CPDQNLearner(
            approximator = RL.NeuralNetworkApproximator(
                model = CPRL.build_model(CPRL.VariableOutputGCNLSTM, fixedGCNargs),
                optimizer = ADAM(0.0005f0)
            ),
            target_approximator = RL.NeuralNetworkApproximator(
                model = CPRL.build_model(CPRL.VariableOutputGCNLSTM, fixedGCNargs),
                optimizer = ADAM(0.0005f0)
            ),
            loss_func = huber_loss,
            stack_size = nothing,
            γ = 0.999f0,
            batch_size = 1,
            update_horizon = 15,
            min_replay_history = 1,
            update_freq = 50,
            target_update_freq = 200,
            seed = 22,
        ), 
        # explorer = CPRL.DirectedExplorer(;
            explorer = CPRL.CPEpsilonGreedyExplorer(
                ϵ_stable = 0.01,
                kind = :linear,
                ϵ_init = 1.0,
                warmup_steps = 0,
                decay_steps = 1000,
                step = 1,
                is_break_tie = false, 
                #is_training = true,
            )
            # direction = ((values, mask) -> view(keys(values), mask)[1]),
            # directed_steps=1000
        # )
    ),
    trajectory = RL.CircularCompactSARTSATrajectory(
        capacity = 3000, 
        state_type = Float32, 
        state_size = state_size,
        action_type = Int,
        action_size = (),
        reward_type = Float32,
        reward_size = (),
        terminal_type = Bool,
        terminal_size = ()
    ),
    role = :DEFAULT_PLAYER
)