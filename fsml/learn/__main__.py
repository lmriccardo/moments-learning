from fsml.learn.train import train
from fsml.learn.test import test
import os
import fsml.learn.config as config


def main() -> None:
    file = os.path.join(os.getcwd(), "data/meanstd/BIOMD00001_MeanStd.csv")
    kwargs = {
        "num_hidden_input"  :config.MLP_NUM_HIDDEN_INPUT,
        "num_hidden_output" :config.MLP_NUM_HIDDEN_OUTPUT,
        "hidden_input_size" :config.MLP_HIDDEN_INPUT_SIZE,
        "hidden_output_size":config.MLP_HIDDEN_OUTPUT_SIZE
    }

    outputs = train(file, **kwargs)
    _ = test(outputs, **kwargs)


def test_better_mlp() -> None:
    import itertools
    file = os.path.join(os.getcwd(), "data/meanstd/BIOMD00003_MeanStd.csv")
    values = [
        config.KF_POSSIBILITIES,
        config.MLP_NUM_HIDDEN_INPUT_POSSIBILITIES,
        config.MLP_NUM_HIDDEN_OUTPUT_POSSIBILITIES
    ]

    current_max_accuracy = 0.0
    current_best_params = (0, 0, 0)
    for kf_split, num_hidden_intput, num_hidden_output in itertools.product(*values):
        kwargs = {
            "num_hidden_input"  : num_hidden_intput,
            "num_hidden_output" : num_hidden_output,
            "hidden_input_size" : config.MLP_HIDDEN_INPUT_SIZE,
            "hidden_output_size": config.MLP_HIDDEN_OUTPUT_SIZE
        }

        outputs = train(file, k_fold=kf_split, **kwargs)
        accs = test(outputs, **kwargs)
        
        if (acc := accs[0]) > current_max_accuracy:
            current_max_accuracy = acc
            current_best_params = (kf_split, kwargs)
    
    print("#" * 100)
    print(f"Final Best Accuracy: {current_max_accuracy}")
    print(f"Final Best Parameters: {current_best_params}")


# main()
# test_better()