using ArgParse
using SeaPearl

include("kidney_exchange_config.jl")

function get_kep_parameters_from_args()
    args = parse_commandline()
    kep_params = KepParameters()
    kep_params.num_episodes = args[:n]
    kep_params.num_restarts_per_instance = args[:r]
    kep_params.eval_timeout = args[:timeout]
    kep_params.num_instances = args[:i]
    kep_params.num_random_heuristics = args[:random]
    kep_params.num_nodes = args[:num_nodes]
    kep_params.num_nodes_eval = args[:num_nodes]
    kep_params.seed = args[:seed]
    kep_params.seed_eval = args[:seed_eval]
    kep_params.density = args[:density]
    kep_params.reward = SeaPearl.GeneralReward
    
    return kep_params
end

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