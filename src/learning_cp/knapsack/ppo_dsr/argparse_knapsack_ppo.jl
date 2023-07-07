import ArgParse.ArgParseSettings
import ArgParse.@add_arg_table
import ArgParse.parse_args
using CSV, DataFrames
using Random

function parse_commandline()
    """
    Parse the command line arguments and return a dictionary containing the values
    """
    s = ArgParseSettings()

    @add_arg_table s begin
        "--random_seed", "-s"
            help = "seed to intialize the random number generator"
            arg_type = Int
            default = 0
            required = false
        "--time_limit", "-t"
            help = "total CPU time (in seconds) allowed"
            arg_type = Int
            default = 1000000
            required = false
        "--csv_path"
            help = "name of the csv file path for saving performance, if not found, nothing is saved"
            arg_type = String
            required = false

        #Knapsack generator settings
        "--nb_items"
            help = "Number of items in knapsack instances"
            arg_type = Int
            default = 10
            required = false
        "--max_weight"
            help = "Maximum weight for items"
            arg_type = Int
            default = 10
            required = false
        "--correlation"
            help = "Maximum weight for items"
            arg_type = Float
            default = 0.2
            required = false
        
        #Experiment settings
        "--num_episodes"
            help = "number of training episodes"
            arg_type = Int
            default = 100
            required = false
        "--eval_freq"
            help = "episode frequency of evaluation during training"
            arg_type = Int
            default = 10
            required = false
        "--num_instances"
            help = "number of instances for evaluation"
            arg_type = Int
            default = 5
            required = false
        "--num_random_heuristics"
            help = "Number of random heuristic for testing"
            arg_type = Int
            default = 0
            required = false

        #PPO settings
        "--gamma"
            help = "Gamma"
            arg_type = Float
            default = 0.99f0
            required = false
        "--lambda"
            help = "Lambda"
            arg_type = Float
            default = 0.95f0
            required = false
        "--clip_range"
            help = "Clip Range (epsilon)"
            arg_type = Float
            default = 0.2f0
            required = false
        "--max_grad_norm"
            help = "Maximum norm for loss gradient"
            arg_type = Float
            default = 0.5f0
            required = false
        "--n_epochs"
            help = "Number of epochs for each update"
            arg_type = Int
            default = 10
            required = false
        "--n_microbatches"
            help = "Number of microbatches for update"
            arg_type = Int
            default = 32
            required = false
        "--actor_loss_weight"
            help = "Weight for actor loss in total loss"
            arg_type = Float
            default = 1.0f0
            required = false
        "--critic_loss_weight"
            help = "Weight for critic loss in total loss"
            arg_type = Int
            default = 0.5f0
            required = false
        "--entropy_loss_weight"
            help = "Weight for entropy loss in total loss"
            arg_type = Int
            default = 0.1f0
            required = false
        "--update_freq"
            help = "Network weight update frequency"
            arg_type = Int
            default = 128
            required = false
        
    end
    return parse_args(s)
end

function set_settings()
    """
    Main function of the script
    """
    parsed_args = parse_commandline()

    random_seed = parsed_args["random_seed"]
    time_limit = parsed_args["time_limit"]
    csv_path = parsed_args["csv_path"]

    nb_items = parsed_args["nb_items"]
    max_weight = parsed_args["max_weight"]
    correlation = parsed_args["correlation"]

    num_episodes = parsed_args["num_episodes"]
    eval_freq = parsed_args["eval_freq"]
    num_instances = parsed_args["num_instances"]
    num_random_heuristics = parsed_args["num_random_heuristics"]

    gamma = parsed_args["gamma"]
    lambda = parsed_args["lambda"]
    clip_range = parsed_args["clip_range"]
    max_grad_norm = parsed_args["max_grad_norm"]
    n_epochs = parsed_args["n_epochs"]
    n_microbatches = parsed_args["n_microbatches"]
    actor_loss_weight = parsed_args["actor_loss_weight"]
    critic_loss_weight = parsed_args["critic_loss_weight"]
    entropy_loss_weight = parsed_args["entropy_loss_weight"]
    update_freq = parsed_args["update_freq"]
    trajectory_capacity = update_freq

    if isnothing(csv_path)
        csv_path = ""
        save_performance = false
    else
        save_performance = true
    end

    # @eval(Base.Sys, CPU_THREADS=$nb_core)

    Random.seed!(random_seed)

    knapsack_generator = SeaPearl.KnapsackGenerator(nb_items, max_weight, correlation)
    experiment_setup = KnapsackExperimentConfig(num_episodes, eval_freq, num_instances, num_random_heuristics)
    knapsack_agent_config = KnapsackPPOAgentConfig(gamma, lambda, clip_range, max_grad_norm, n_epochs, n_microbatches, actor_loss_weight, critic_loss_weight, entropy_loss_weight, output_size, update_freq, trajectory_capacity)

    return knapsack_generator, experiment_setup, knapsack_agent_config, csv_path, save_model
end
