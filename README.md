# First and Second-order moments Learning

FSML is the result of a project for the course *Automatic Verification Of Intelligent Systems* at the University of Rome "La Sapienza" A.A. 2022/2023. The first goal of this project is to create a simple Neural Network that is able to learn how to predict the *mean* and the *variance* from the fixed parameters of a time series model. The second objective is to experiment if it is also possible the inverse. 

A more detailed version that explain the main reasons and the entire procedure that I setted up to complete the project can be found in the PDF report `report.pdf` in the root folder. 

In this README I will just give to you a quick overview of the project and its technical details.

## The Project Idea

In time series analysis and modeling the underlying properties and behaviours of a time series are very important to be understood. These are described mainly by some factors like: the mean, the variance, the covariance, the auto and partial auto-correlation coefficients. In this case I was focusing only on the first two moments that describes a time series, in particular in the relationship that they have with respect to the parameters of the model. 

In general this relationship depends on the structure of the model, on the nature (if determinstic or stochastic), on the initial conditions (the initial values) and on the parameters themselves. Once we have a deterministic model and we are able to fix the initial conditions, by only changing the parameters we could try to find a way either to: come up with a well-formed formula that describes that relationship, or just implement a learning model and experiment with it. 

In this case I decided to implement a simple *Deep Feed-Forward Neural Network* with: 1 input linear layer, 4 hidden linear layer with 50 neurons each, 2 hidden layer with 30 neurons each and 1 output layer. Obviously the number of neurons for the input and the output layer depends on the size of the problem, i.e., the number of parameters and twice the number of outputs of the model. For the training I have used the simple MSE minimization problem using the **Adam** algorithm with some adjustment like: *Cross Fold Validation*, *Adaptive Learning Rate*, *Regularization* and *Data Pre-processing* (standardization). Further details can be found in the PDF report. 

In the second stage of the project I decided to try also if the inverse relation holds, i.e., from means and variances to parameters. This kind of problems seemed to be more difficult with respect to the previous one, due to the presence of outliers in the output distribution and due also to the different, in order of magnitude, between the input and the outputs. For this reason, insted of applying transformations like the log one, or applying IQR to eliminate the outliers, I decided to use a more robust approach like *Random Forest Regression*. Again, for further details look for the report. 

## The Project Structure

In this part I'm going to describe the most importan files in the project. 

```bash
.
├── fsml                       # Contains the source code of the project
|   ├── learn                  # Contains the source code of the learning part
│   │   ├── data_management    # Contains the source code for data management
│   │   │   ├── dataloader.py  # Code for the dataloader
│   │   │   └── dataset.py     # Code for the dataset   
│   │   ├── models             # Contains the source code with the models (NNs)
│   │   │   └── mlp.py         # Code for the MLP predictor
│   │   ├── config.py          # The configuration for the learning part
│   │   ├── reverse.py         # Code for the reversed problem
│   │   ├── test.py            # Code for testing the models
│   │   └── train.py           # Code for training the models   
│   ├── simulate               # Contains the source code for simulating the models
│   │   ├── simulate.py        # Code for simulating the biomodels
│   │   └── transform.py       # Code for download and transform biological models   
|   └── utils.py               # Just some utilities
├── README.md                  # The README
├── LICENSE.md                 # The license
└── report.pdf                 # The PDF report
```

## Installation and Requirements

There is no kind of procedure for installing the project. Just clone the repo using the usual

```
git clone https://github.com/lmriccardo/moments-learning.git
```

command, and then add the root folder of the FSML package to the PYTHONPATH environment variable. Until now, there is no PiPy way available to install the module. Finally, to be able to import or runs the project you will need to install these Python packages:

- **PyTorch**, for modeling the NN, the dataset and the dataloader
- **Basico**, to use COPASI simulations and dowload the models
- **LibSBML**, to modify the SBML
- **Numpy**, Array math utilities
- **Matplotlib**, to plot some useful graphs
- **Pandas**, handle CSV files
- **Scikit-Learn**, for modeling, train and test Random Forests

All the packages are contained in the `requirements.txt` file in the root folder of the project. Then, just create a new environment and finally install those requirements.

```bash
# Classic Python way
$ python -m venv venv && source venv/bin/activate
$ pip install -r requirements.txt

# Using Conda
$ conda create -n venv python=3.10.8
$ conda activate venv
$ pip install -r requirements.txt
``` 

## Usage

It is possible to use the project module in two different ways. Before proceeding with more technical details, the project is setup into two steps: first step to transform and simulate a model, second step for the learning stage. 

### 1. Command-Line Usage

First way, using the command line. To transform and simulate it is possible to take the availability of the `fsml.simulate` module using the following command `python -m fsml.simulate --help` and obtaining the following output.

```
usage: __main__.py [-h] -m MODEL [-s SIMS] [-l LOG] [-o OUTPUT] [-d DATA] [-p TEST]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The ID of the BioModel wants to be simulated (default: None)
  -s SIMS, --sims SIMS  The total number of simulations to be executed for that model (default: 2500)
  -l LOG, --log LOG     The relative or absolute path to the log folder (default: ./log/)
  -o OUTPUT, --output OUTPUT
                        The relative or absolute path to the output folder (default: ./runs/)
  -d DATA, --data DATA  The relative or absolute path to the data folder (default: ./data/)
  -p TEST, --test TEST  The relative or absolute path to the test folder (default: ./tests/)
```

An example could be

```
$ python -m fsml.simulate -m 1 -s 1000
```

In this case the BioModel [BIOMD0000000001](https://www.ebi.ac.uk/biomodels/BIOMD0000000001) (or Edelstein1996 - EPSP ACh event) would be simulated 1000 times (each time with different parameters). The result of the simulation would be a CSV file stored in the `DATA` folder under the inner folder `meanstd`. The SBML model would be saved into the `TEST` folder, while all the parameters and the initial values for the species inside the `LOG` folder. Finally, the `OUTPUT` folder is just a tmp folder where the files containing the report produced by COPASI are stored. 

The second stage of the project is actually learn the relationship, i.e., train and test the neural network both for the original and the inverse problem. To do this, it is possible to use the `fsml.learn` module. Given the classic help command `python -m fsml.learn --help` this is the output

```
usage: __main__.py [-h] -d DATA [-r] [--batches BATCHES] [--lr LR] [--epochs EPOCHS] [--cv CV] 
                   [--acc-threshold ACC_THRESHOLD] [--grid-search] [--random-search]

options:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  The path to the CSV file or a Data Folder with multiple CSVs (default: None)
  -r, --reverse         True, train the RF for the inverse problem. False otherwise (default: False)
  --batches BATCHES     The number of batches (only for original problem) (default: 32)
  --lr LR               The learning rate (only for original problem) (default: 0.0001)
  --epochs EPOCHS       The total number of epochs (only for original problem) (default: 250)
  --cv CV               The total number of cross validation (default: 3)
  --acc-threshold ACC_THRESHOLD
                        The accuracy threshold (only for original problem) (default: 2.0)
  --grid-search         Turn the Grid Search for Random Forest to True (inverse problem) (default: False)
  --random-search       Turn the Random Search for Random Forest to True (inverse problem) (default: False)
```

Note that some options only affect the original or inverse problem, while some others both like the number of cross validations. An example of running the learning process for the original problem could be

```
$ python -m fsml.learn -d ./data/ --batches 32 --lr 0.001 --epochs 100 --cv 0
```

In this example the command will run the learning procedure (train + test) for all the dataset inside the `./data` folder (note that all the dataset must be CSV files as described in the report), using 32 batches in the dataloader, an initial learning rate of 0.001, running for 100 epochs and finally without using any kind of Cross Validation. Whenever the `--reverse` flag is active then the inverse problem will be solved. Here is an example.

```
$ python -m fsml.learn -d ./data/ --reverse --grid-search --cv 3
```

In this case the inverse problem will be solved for each dataset in the input data folder using a Random Forest Regression method. Moreover, a grid search hyper-parameter tuning method is using with 3 Cross Validation. 

All the configurations for both the original and the inverse problems can be found in the configuration file [`config.py`](./fsml/learn/config.py). If you wants to change some configuration you can either directly modify the configuration file, or use the *Python Script* method. 

### 2. Python Module Usage

Second way, from the PIE (Python Interactive Environment) or a Python Script. This is an example to download, transform and simulate a single model multiple times.

```python
from fsml.simulate.main import transform_and_simulate_one
import fsml.utils as utils
import os.path as opath
import os

# Define the output folders
log_dir = opath.join(os.getcwd(), "log/")
output_dir = opath.join(os.getcwd(), "runs/")
data_dir = opath.join(os.getcwd(), "data/")
test_dir = opath.join(os.getcwd(), "tests/")

# Define the model ID and the number of simulations
model_id = 1
number_of_simulations = 1000

# Setup the seed
utils.setup_seed()

# Run the procedure
transform_and_simulate_one(prefix_path=test_dir,
                           log_dir=log_dir,
                           output_dir=output_dir,
                           data_dir=data_dir,
                           model_id=model_id,
                           nsim=number_of_simulations,
                           job_id=0,
                           gen_do=False)
```

This is an example to train and test the FSML learning model on a folder of dataset for the original problem

```python
import torch.nn as nn

# USE fsml_ prefix to avoid conflict
from fsml.learn.train import train as fsml_train
from fsml.leran.test import test as fsml_test
import fsml.learn.config as fsml_config
import os.path as opath
import os

# Get the folder with all the datasets
data = opath.join(os.getcwd(), "data/")

# Define some arguments
kwargs = {
    "num_hidden_input"   : fsml_config.MLP_NUM_HIDDEN_INPUT,
    "num_hidden_output"  : fsml_config.MLP_NUM_HIDDEN_OUTPUT,
    "hidden_input_size"  : fsml_config.MLP_HIDDEN_INPUT_SIZE,
    "hidden_output_size" : fsml_config.MLP_HIDDEN_OUTPUT_SIZE
}

# Learn the train and retrieve the model path and
# and the dataloader for each trained model
outputs = fsml_train(data, criterion=nn.L1Loss,
                           batch_size=32,
                           k_fold=fsml_config.KF_SPLIT,
                           num_epochs=100,
                           patience=5,
                           **kwargs)

# Then just test
_ = fsml_test(outputs, **kwargs)
```

Finally, this is an example to solve the inverse problem with some changed configuration

```python
from fsml.learn.reverse import train_and_test as fsml_train_and_test
from fsml.utils import count_folder_elements
import fsml.learn.config as fsml_config
import os.path as opath
import time

# Changing the number of CV for the random search
global fsml_config.RAND_SEARCH_NUM_CROSS_VALIDATION
fsml_config.RAND_SEARCH_NUM_CROSS_VALIDATION = 5

# Changing the Random Search Boostrap possibilities to [True]
global fsml_config.RAND_SEARCH_BOOSTRAP
fsml_config.RAND_SEARCH_BOOSTRAP = [True]

# We can do this for all possibile configurations ... 

# Retrieve all the CSV files from the default Data Path
condition = lambda x: x.endswith(".csv")
_, files = count_folder_elements(config.DATA_PATH, condition)

# Iterate through all the files and run the learning procedure
for dataset_file in files:
    start_time = time.time()
    print(f":-----------: Solving reverse problem for {dataset_file} :-----------:")

    # Launch the procedure
    trained_model = fsml_train_and_test(dataset_file,
                                        random_search=True,
                                        grid_search=True)

    final_time = time.time() - start_time
    print(f"Process Ended in {final_time:0.3f} seconds")
```

### 3. Final Notes on Usability

Obviously there are many way in which a possible user can use this module. In all the previous examples I decided to show how to use some built-in modules like `transform_and_simulate_one` to transform and simulate, in the way I decided, a biomodel, or the `train` and `test` functions in the learning stage (for the original problem). 

In general, while using the command line utilities my goal is to make the user to be able to run the same experiments that I did, with the FSML module in a Python Script the user can do whatever he wants and decide its own workflow. For example, he can creates a custom `train` and `test` function using directly the `FSMLOneMeanStdDataset` or the `FSMLMeanStdDataset`, the `FSMLDataLoader` that are contained in the `fsml.learn.data_management` package. For more informations about the inner class and functions you can directly read the module documentation. 