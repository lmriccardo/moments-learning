from fsml.transform import convert_one
from fsml.simulate import run_one
from multiprocessing import Pool
import fsml.utils as utils
from typing import Dict
import os.path as opath
import argparse
import time
import os


# -----------------------------------------------------------------------------
# Transform and Simulate
# -----------------------------------------------------------------------------

def transform_and_simulate_one(kwargs: Dict[str, any]) -> str:
    # Parameters for transformation
    prefix_path = kwargs["prefix_path"]
    model_id    = kwargs["model_id"]

    # Parameters for simulating
    log_dir    = kwargs["log_dir"]
    output_dir = kwargs["output_dir"]
    data_dir   = kwargs["data_dir"]
    job_id     = kwargs["job_id"]
    nsim       = kwargs["nsim"]

    # Transform
    trans_model_path = convert_one(prefix_path, model_id)

    # Simulate
    run_one(trans_model_path, log_dir, output_dir, data_dir, job_id, nsim)

    return trans_model_path


def __transform_and_simulate(**kwargs: Dict[str, any]) -> None:
    # Parameters for transformation
    prefix_path = kwargs["prefix_path"]
    nmodels     = kwargs["nmodels"]
    paths_file  = kwargs["paths_file"]

    # Parameters for simulation
    log_dir        = kwargs["log_dir"]
    output_dir     = kwargs["output_dir"]
    data_dir       = kwargs["data_dir"]
    nsim_per_model = kwargs["nsim_per_model"]

    cpu_count = os.cpu_count()
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
                               "nsim"        : nsim_per_model
                          },
                range(minc, maxc)
            ))

            trans_model_path = pool.map(transform_and_simulate_one, args)
            output_paths += trans_model_path
    
    utils.write_paths(output_paths, paths_file)
    utils.remove_original(output_paths)
    print(f"----------- [*] PROCEDURE ENDED IN: {time.time() - start_timer} sec -----------")


def main() -> None:
    # argument_parser = argparse.ArgumentParser()
    # argument_parser.add_argument("simulate", help="Flag that if sets to True then ")

    log_dir     = opath.join(os.getcwd(), "log")
    output_dir  = opath.join(os.getcwd(), "runs")
    data_dir    = opath.join(os.getcwd(), "data/simulations/")
    prefix_path = opath.join(os.getcwd(), "tests")
    paths_file  = opath.join(os.getcwd(), "data/paths.txt")

    nmodels = 5
    nsim_per_model = 10

    utils.setup_seed()
    
    __transform_and_simulate(
        prefix_path = prefix_path,
        nmodels = nmodels,
        paths_file = paths_file,
        log_dir = log_dir,
        output_dir = output_dir,
        data_dir = data_dir,
        nsim_per_model = nsim_per_model
    )