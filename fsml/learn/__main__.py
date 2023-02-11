from fsml.learn.train import train
from fsml.learn.test import test
from fsml.learn.reverse import train_and_test
from fsml.utils import count_folder_elements
import os
import fsml.learn.config as config
import time
import argparse


def learn() -> None:
    file = os.path.join(os.getcwd(), "data/meanstd/")
    kwargs = {
        "num_hidden_input"  :config.MLP_NUM_HIDDEN_INPUT,
        "num_hidden_output" :config.MLP_NUM_HIDDEN_OUTPUT,
        "hidden_input_size" :config.MLP_HIDDEN_INPUT_SIZE,
        "hidden_output_size":config.MLP_HIDDEN_OUTPUT_SIZE
    }

    outputs = train(file, **kwargs)
    _ = test(outputs, **kwargs)


def learn_reverse() -> None:
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


def main() -> None:
    argument_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argument_parser.add_argument('-d', '--data',
                                 help="The path to the CSV file or a Data Folder with multiple CSVs",
                                 required=True,
                                 type=str)
    argument_parser.add_argument('-r', "--reverse",
                                 help="True, train the RF for the inverse problem. False otherwise",
                                 required=False,
                                 action="store_true")
    argument_parser.add_argument("--batches",
                                 help="The number of batches (only for original problem)",
                                 required=False,
                                 default=config.BATCH_SIZE)
    argument_parser.add_argument("--lr",
                                 help="The learning rate (only for original problem)",
                                 required=False,
                                 default=config.LR)
    argument_parser.add_argument("--epochs",
                                 help="The total number of epochs (only for original problem)",
                                 required=False,
                                 default=config.NUM_EPOCHS)
    argument_parser.add_argument("--cv",
                                 help="The total number of cross validation",
                                 required=False,
                                 default=config.KF_SPLIT)
    argument_parser.add_argument("--acc-threshold",
                                 help="The accuracy threshold (only for original problem)",
                                 required=False,
                                 default=config.ACCURACY_THRESHOLD)
    argument_parser.add_argument("--grid-search",
                                 help="Turn the Grid Search for Random Forest to True (inverse problem)",
                                 required=False,
                                 action="store_true")
    argument_parser.add_argument("--random-search",
                                 help="Turn the Random Search for Random Forest to True (inverse problem)",
                                 required=False,
                                 action="store_true")
    
    args = argument_parser.parse_args()

    print(args.data)

main()