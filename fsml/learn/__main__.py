from fsml.learn.train import train
from fsml.learn.test import test
from fsml.learn.reverse import train_and_test
from fsml.utils import count_folder_elements
import os
import fsml.learn.config as config
import time


def main() -> None:
    file = os.path.join(os.getcwd(), "data/meanstd/")
    kwargs = {
        "num_hidden_input"  :config.MLP_NUM_HIDDEN_INPUT,
        "num_hidden_output" :config.MLP_NUM_HIDDEN_OUTPUT,
        "hidden_input_size" :config.MLP_HIDDEN_INPUT_SIZE,
        "hidden_output_size":config.MLP_HIDDEN_OUTPUT_SIZE
    }

    outputs = train(file, **kwargs)
    _ = test(outputs, **kwargs)


def main_reverse() -> None:
    # condition = lambda x: x.endswith('.csv')
    # _, files = count_folder_elements(config.DATA_PATH, condition)
    
    files = [
        os.path.join(config.DATA_PATH, "BIOMD00005_MeanStd.csv"),
        os.path.join(config.DATA_PATH, "BIOMD00007_MeanStd.csv")
    ]
    
    for file in files:
        start_time = time.time()
        print("-" * 50 + " " + file + " " + "-" * 50)
        model = train_and_test(file, grid_search=True)
        print(f"[*] Model saved at {model}")
        
        final_time = time.time() - start_time
        print(f"[*] Process ended in {final_time:0.3f} seconds")


#main()
main_reverse()