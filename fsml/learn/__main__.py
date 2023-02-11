from fsml.learn.train import train
from fsml.learn.test import test
from fsml.learn.reverse import train_and_test
from fsml.utils import count_folder_elements
import os
import fsml.learn.config as config
import time
import argparse


def learn(data_file    : str, 
          num_batch    : int, 
          num_epochs   : int, 
          lr           : float,
          cv           : int, 
          acc_threshold: float
) -> None:
    kwargs = {
        "num_hidden_input"  :config.MLP_NUM_HIDDEN_INPUT,
        "num_hidden_output" :config.MLP_NUM_HIDDEN_OUTPUT,
        "hidden_input_size" :config.MLP_HIDDEN_INPUT_SIZE,
        "hidden_output_size":config.MLP_HIDDEN_OUTPUT_SIZE
    }

    outputs = train(data_file, 
                    batch_size=num_batch, 
                    num_epochs=num_epochs,
                    lr=lr,
                    k_fold=cv,
                    accuracy_threshold=acc_threshold,
                    **kwargs)
    
    _ = test(outputs, **kwargs)


def learn_reverse(data_file: str, grid_search: bool, random_search: bool, cv: int) -> None:
    if os.path.isfile(data_file):
        files = [data_file]
    else:
        condition = lambda x: x.endswith('.csv')
        _, files = count_folder_elements(data_file, condition)

    config.RAND_SEARCH_NUM_CROSS_VALIDATION = cv
    config.GRID_SEARCH_NUM_CROSS_VALIDATION = cv
    
    for file in files:
        start_time = time.time()
        print("-" * 50 + " " + file + " " + "-" * 50)
        model = train_and_test(file, grid_search=grid_search, random_search=random_search)
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

    data_path     = os.path.abspath(args.data)
    reverse       = args.reverse
    batches       = args.batches
    lr            = args.lr
    epochs        = args.epochs
    cv            = args.cv
    acc_threshold = args.acc_threshold
    grid_search   = args.grid_search
    random_search = args.random_search

    # If not reverse then the original problem must be solved
    if not reverse:
        learn(data_path, batches, epochs, lr, cv, acc_threshold)
        return
    
    # Otherwise solve for the reverse problem
    learn_reverse(data_path, grid_search, random_search, cv)


main()