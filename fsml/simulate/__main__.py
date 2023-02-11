from fsml.simulate.main import transform_and_simulate, \
                               transform_and_simulate_one
import fsml.utils as utils
import argparse
import os.path as opath
import os


def main() -> None:
    argument_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argument_parser.add_argument('-m', "--model", 
                                 help="The ID of the BioModel wants to be simulated", 
                                 required=True,
                                 type=int)
    argument_parser.add_argument('-s', '--sims',
                                 help="The total number of simulations to be executed for that model",
                                 required=False,
                                 default=2500,
                                 type=int)
    argument_parser.add_argument('-l', "--log",
                                 help="The relative or absolute path to the log folder",
                                 required=False,
                                 default=opath.join(os.getcwd(), "log/"),
                                 type=str)
    argument_parser.add_argument('-o', "--output",
                                 help="The relative or absolute path to the output folder",
                                 required=False,
                                 default=opath.join(os.getcwd(), "runs/"),
                                 type=str)
    argument_parser.add_argument('-d', "--data",
                                 help="The relative or absolute path to the data folder",
                                 required=False,
                                 default=opath.join(os.getcwd(), "data/"),
                                 type=str)
    argument_parser.add_argument('-p', "--test",
                                 help="The relative or absolute path to the test folder",
                                 required=False,
                                 default=opath.join(os.getcwd(), "tests/"),
                                 type=str)
    
    args = argument_parser.parse_args()
    

    log_dir        = args.log
    output_dir     = args.output
    data_dir       = args.data
    prefix_path    = args.test
    nsim_per_model = args.sims
    model_id       = args.model

    utils.setup_seed()

    kwargs = {
        "prefix_path": prefix_path,
        "model_id"   : model_id,
        "log_dir"    : log_dir,
        "output_dir" : output_dir,
        "data_dir"   : data_dir,
        "job_id"     : 0,
        "nsim"       : nsim_per_model,
        "gen_do"     : False
    }

    transform_and_simulate_one(kwargs)


main()