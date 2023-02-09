from fsml.simulate.transform import convert_one
from fsml.simulate import run_one
from multiprocessing import Pool
import fsml.utils as utils
from typing import Dict
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
    gen_do     = kwargs["gen_do"]

    # Transform
    trans_model_path = convert_one(prefix_path, model_id)

    # Simulate
    run_one(trans_model_path, log_dir, output_dir, data_dir, job_id, nsim, gen_do)

    return trans_model_path


def transform_and_simulate(**kwargs: Dict[str, any]) -> None:
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