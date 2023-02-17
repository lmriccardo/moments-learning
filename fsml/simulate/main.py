from fsml.simulate.transform import convert_one
from fsml.simulate.simulate import run_one
from multiprocessing import Pool
import fsml.utils as utils
from typing import Dict
import time


# -----------------------------------------------------------------------------
# Transform and Simulate
# -----------------------------------------------------------------------------

def transform_and_simulate_one(kwargs: Dict[str, any]) -> str:
    """
    Given a kind of configuration, it download, transforms and simulate a biomodel.

    :param kwargs: A Python Dictionary that must contains the following fields:
                   - prefix_path, with the path of the test folder
                   - model_id, with the ID of the model to simulate
                   - log_dir, with the path of the log folder
                   - output_dir, with the path of the output folder
                   - data_dir, with the path of the data folder
                   - job_id, not important left to 0
                   - nsim, the number of simulations
                   - gen_do, boolean True generate also the dense output

    :return: the path of the model stored in the test folder
    """
    # Parameters for transformation
    prefix_path = kwargs["prefix_path"]
    model_id    = kwargs["model_id"]

    # Parameters for simulating
    log_dir    = kwargs["log_dir"]
    output_dir = kwargs["output_dir"]
    data_dir   = kwargs["data_dir"]
    job_id     = kwargs["job_id"]
    nsim       = kwargs["nsim"]
    gen_do     = kwargs["gen_do"]

    # Transform
    trans_model_path = convert_one(prefix_path, model_id)

    # Simulate
    run_one(trans_model_path, log_dir, output_dir, data_dir, job_id, nsim, gen_do)

    return trans_model_path


def transform_and_simulate(**kwargs: Dict[str, any]) -> None:
    """
    Download, transform and simulate a bunch of model in a multiprocessing way.

    :param kwargs: A Python Dictionary that must contains the following fields:
                   - prefix_path, with the path of the test folder
                   - nmodels, the number of models to download, transform and simulate
                   - paths_file, the path to the file where to store the final paths
                   - log_dir, with the path of the log folder
                   - output_dir, with the path of the output folder
                   - data_dir, with the path of the data folder
                   - nsim_per_model, the number of simulations to run for each model
                   - gen_do, boolean True generate also the dense output
    
    :return:
    """
    # Parameters for transformation
    prefix_path = kwargs["prefix_path"]
    nmodels     = kwargs["nmodels"]
    paths_file  = kwargs["paths_file"]

    # Parameters for simulation
    log_dir         = kwargs["log_dir"]
    output_dir      = kwargs["output_dir"]
    data_dir        = kwargs["data_dir"]
    nsim_per_model  = kwargs["nsim_per_model"]
    gen_denseoutput = kwargs["gen_do"]

    cpu_count = 2
    output_paths = []

    print(f"----------- [*] STARTING PROCEDURE OF DOWNLOADING AND SIMULATION OF: {nmodels} MODELS -----------")
    start_timer = time.time()

    for (minc, maxc) in utils.get_ranges(nmodels, cpu_count):
        with Pool(cpu_count) as pool:
            args = list(
                map(
                lambda x: {
                               "prefix_path" : prefix_path,
                               "model_id"    : x,
                               "log_dir"     : log_dir,
                               "output_dir"  : output_dir,
                               "data_dir"    : data_dir,
                               "job_id"      : x,
                               "nsim"        : nsim_per_model,
                               "gen_do"      : gen_denseoutput
                          },
                range(minc, maxc)
            ))

            trans_model_path = pool.map(transform_and_simulate_one, args)
            output_paths += trans_model_path
    
    utils.write_paths(output_paths, paths_file)
    utils.remove_original(output_paths)
    print(f"----------- [*] PROCEDURE ENDED IN: {time.time() - start_timer} sec -----------")