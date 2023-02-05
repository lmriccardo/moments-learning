from fsml.main import transform_and_simulate, transform_and_simulate_one
import fsml.utils as utils
import os.path as opath
import os


def main() -> None:
    # argument_parser = argparse.ArgumentParser()
    # argument_parser.add_argument("simulate", help="Flag that if sets to True then ")

    log_dir     = opath.join(os.getcwd(), "log")
    output_dir  = opath.join(os.getcwd(), "runs")
    data_dir    = opath.join(os.getcwd(), "data/")
    prefix_path = opath.join(os.getcwd(), "tests")
    paths_file  = opath.join(os.getcwd(), "data/paths.txt")

    nmodels = 0
    nsim_per_model = 2500

    utils.setup_seed()
    
    # transform_and_simulate(
    #     prefix_path = prefix_path,
    #     nmodels = nmodels,
    #     paths_file = paths_file,
    #     log_dir = log_dir,
    #     output_dir = output_dir,
    #     data_dir = data_dir,
    #     nsim_per_model = nsim_per_model,
    #     gen_do = False
    # )

    model_id = 8

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