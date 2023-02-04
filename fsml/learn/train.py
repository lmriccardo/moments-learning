import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss, _WeightedLoss

import fsml.utils as utils
from fsml.learn.data_mangement.dataset import FSMLOneMeanStdDataset,  \
                                              get_dataset_by_indices, \
                                              FSMLMeanStdDataset

from fsml.learn.data_mangement.dataloader import FSMLDataLoader
from sklearn.model_selection import KFold
import fsml.learn.models.nets as nets
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional
from tqdm import tqdm
from functools import wraps
import os
import os.path as opath
import time
    

class KFoldCrossValidationWrapper:
    r""" A simple wrapper class for K-Fold Cross Validation """

    @staticmethod
    def setup_kFold_validation(dataset    : FSMLOneMeanStdDataset, 
                               kf_split   : int, 
                               batch_size : int) -> List[Tuple[int, FSMLDataLoader]]:
        r"""
        Setup the kfold validation, i.e., returns a list of
        triple (fold index, train dataloader, validation dataloader)

        :param dataset: The dataset to split
        :param kf_split: The total number of split
        :param batch_size: the batch_size argument to the dataloader
        :return: a list of triple (fold index, train dataloader, validation dataloader)
        """
        # Create the splitter and the dataloader list
        kfold_splitter = KFold(n_splits=kf_split, shuffle=True)
        tt_list = []

        for fold_num, (train_ids, test_ids) in enumerate(kfold_splitter.split(dataset)):
            fold_dataset = get_dataset_by_indices(dataset, train_ids, test_ids)
            fold_dataloader = FSMLDataLoader(fold_dataset, batch_size=batch_size)
            tt_list.append((fold_num, fold_dataloader))

        return tt_list
    
    @staticmethod
    def kFoldValidation(dataset    : FSMLOneMeanStdDataset,
                        model      : nets.FSML_MLP_Predictor,
                        epoch      : int,
                        kf_split   : int,
                        batch_size : int,
                        use        : bool=True) -> Callable:
        r""" Run kFold Cross Validation """
        def _kFoldValidation(func):
            
            @wraps(func)
            def wrapper(*args, **kwargs) -> Tuple[float, float]:
                """ The wrapper that returns the train and test total loss """
                # If use is set to False then just run the train set and returns
                if not use:
                    train_loss, test_acc = func()
                    return train_loss, test_acc
                
                # Set up the KFold validation and returns the dataloaders
                tt_list = KFoldCrossValidationWrapper.setup_kFold_validation(
                    dataset, kf_split, batch_size
                )

                progress_bar = tqdm(tt_list, desc=f"Epoch: {epoch} -- ", leave=True, position=0)
                fold_total_train_loss = 0.0
                fold_total_train_accuracy = 0.0
                for fold_num, fold_dataloader in progress_bar:
                    
                    # Create a new progress bar for the train step
                    dl_progress_bar = tqdm(fold_dataloader(), desc=f"Fold: {fold_num} -- ", leave=False, position=1)

                    # Get the final train loss
                    fold_train_loss, _ = func(fold_dataloader, dl_progress_bar)
                    fold_total_train_loss += fold_train_loss

                    # Set the dataset for testing
                    fold_dataloader.dataset.test()

                    # Validate
                    total_accuracy = 0.0
                    with torch.no_grad():
                        model.eval()  # Set the model for evaluation
                        for i_batch, (test_x, test_y, _) in enumerate(fold_dataloader):
                            current_accuracy = utils.compute_accuracy(model(test_x), test_y)
                            total_accuracy += current_accuracy
                        
                    total_accuracy /= (i_batch + 1)
                    progress_bar.set_postfix_str(f"Train Loss: {fold_train_loss}, Test Acc: {total_accuracy}")
                    progress_bar.refresh()

                    fold_total_train_accuracy += total_accuracy

                fold_total_train_loss /= (fold_num + 1)
                fold_total_train_accuracy /= (fold_num + 1)

                progress_bar.set_description_str(f"Epoch: {epoch} -- ")
                progress_bar.set_postfix_str(
                    f"Train Loss: {fold_total_train_loss}, Train Acc: {fold_total_train_accuracy}")
                progress_bar.refresh()

                return fold_total_train_loss, fold_total_train_accuracy
            
            return wrapper
        
        return _kFoldValidation


class Trainer:
    def __init__(self, train_dataset      : FSMLOneMeanStdDataset,              # The input train dataset
                       train_dataloader   : FSMLDataLoader,                     # The input train dataloader
                       optimizer          : optim.Optimizer,                    # The optimizer to use
                       model              : nn.Module,                          # The predictor
                       num_epochs         : int                   = 30,         # Total number of epochs
                       criterion          : _Loss | _WeightedLoss = nn.MSELoss, # The Loss function to be used
                       num_hidden_input   : int                   = 5,          # Number of hidden layer input side
                       num_hidden_output  : int                   = 5,          # Number of hidden layers output side
                       hidden_input_size  : int                   = 50,         # Size of hidden input layers
                       hidden_output_size : int                   = 30,         # Size of hidden output layers
                       k_fold             : int                   = 5,          # The number of fold for KFoldCrossValidation
                       accuracy_threshold : float                 = 0.94,       # Stop for accuracy grater than this

                       model_path         : str = os.path.join(os.getcwd(), "models")   # The path where to store the models
    ) -> None:
        """
        :param train_dataset: The input train dataset
        :param train_dataloader: The input train dataloader
        :param optimizer: The optimizer to use
        :param model: The predictor
        :param num_epochs: Total number of epochs
        :param criterion: The Loss function to be used
        :param num_hidden_input: Number of hidden layer input side
        :param num_hidden_output: Number of hidden layers output side
        :param hidden_input_size: Size of hidden input layers
        :param hidden_output_size: Size of hidden output layers
        :param model_path: The path where to store the models
        :param k_fold: number of KFold Cross Validation runs
        :param accuracy_threshold: Stop when the current accuracy overcome a value
        """
        self.train_dataset      = train_dataset
        self.train_dataloder    = train_dataloader
        self.optimizer          = optimizer
        self.num_epochs         = num_epochs
        self.criterion          = criterion()
        self.num_hidden_input   = num_hidden_input
        self.num_hidden_output  = num_hidden_output
        self.hidden_input_size  = hidden_input_size
        self.hidden_output_size = hidden_output_size
        self.model_path         = model_path
        self.model              = model
        self.k_fold             = k_fold
        self.use_kfold          = (k_fold != 0)
        self.accuracy_threshold = accuracy_threshold

        self.train_losses = []
        self.train_accs   = []

    def _run_epoch(self, epoch: int) -> Tuple[float, float]:
        """ Run one single epoch of training """

        @KFoldCrossValidationWrapper.kFoldValidation(
            self.train_dataset, self.model, epoch, self.k_fold, self.train_dataloder.batch_size, self.use_kfold)
        def __run_epoch(
            dataloader: Optional[FSMLDataLoader]=None, prog_bar: Optional[tqdm]=None
        ) -> Tuple[float, float]:
            # If the input dataloader is None then use the default one
            # The default dataloader is the one already defined in the class
            if not dataloader:
                dataloader = self.train_dataloder

            if not prog_bar:
                prog_bar = tqdm(dataloader(), desc=f"Epoch: {epoch} -- ", leave=True)
            
            train_loss = torch.zeros(1)
            train_acc  = 0.0
            train_step = 0
            self.optimizer.zero_grad()
            for _, (train_x, train_y, _) in enumerate(prog_bar):
                output      = self.model(train_x)
                loss        = self.criterion(output, train_y)
                train_loss += loss

                with torch.no_grad():
                    acc = utils.compute_accuracy(output, train_y)
                    train_acc += acc
                
                loss.backward()
                self.optimizer.step()

                train_step += 1
                prog_bar.set_postfix_str(f"Train Loss: {(train_loss / train_step).item()}")
            
            train_loss /= train_step
            train_acc  /= train_step
            
            return train_loss.item(), train_acc
        
        return __run_epoch(self.train_dataloder)
    
    def run(self) -> None:
        """ Run the training """
        # Set the model for training
        self.model.train()

        print("[*] Start training the model")
        start_time = time.time()
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._run_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            if train_acc > self.accuracy_threshold:
                print(f"[*] Accuracy threshold reached. Stop")
                self.num_epochs = epoch + 1
                break
        
        end_time = time.time()
        print(f"[*] Training procedure ended in {end_time - start_time} msec")

        filepath_linux_format = opath.basename(self.train_dataset.csv_file).replace('\\', '/')
        csv_filename = opath.basename(filepath_linux_format)
        model_filepath = opath.join(
            self.model_path, 
            f"{csv_filename}_{self.model.__class__.__name__}.pth"
        )
        print(f"[*] Saving the model {model_filepath}")
        torch.save(self.model.state_dict(), model_filepath)
    
    def plot(self) -> None:
        """ Plot the result (train loss and train acc over epochs) """
        epochs = list(range(self.num_epochs))
        plt.plot(epochs, self.train_losses, label="Train losses")
        plt.plot(epochs, self.train_accs, label="Train accuracies")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Train losses and Accuracies")
        plt.legend(loc="upper right")
        plt.show()


def __train_one(train_dataset     : FSMLOneMeanStdDataset,
                criterion         : _Loss | _WeightedLoss,
                batch_size        : int,
                k_fold            : int,
                num_epochs        : int,
                num_hidden_input  : int,
                num_hidden_output : int,
                hidden_input_size : int,
                hidden_output_size: int,
                accuracy_threshold: float) -> None:
    """ Run one training with the input dataset and configuration """
    # Log the dataset for the training
    print("[*] Called training procedure with dataset")
    print(train_dataset)

    print("[*] Creating the dataloader")
    train_dataloader = FSMLDataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, drop_last=True
    )

    print("[*] Instantiating the Predictor")
    predictor = nets.FSML_MLP_Predictor(
        train_dataset.input_size,  num_hidden_input,  hidden_input_size,
        train_dataset.output_size, num_hidden_output, hidden_output_size
    )
    print(predictor)

    print("[*] Creating the Adam Optimizer")
    optimizer = optim.Adam(predictor.parameters(), lr=0.0001)
    print(optimizer)

    print("[*] Instantiating the Trainer")
    trainer = Trainer(
        train_dataset, train_dataloader,
        optimizer, predictor,
        num_epochs, criterion,
        num_hidden_input, num_hidden_output,
        hidden_input_size, hidden_output_size, 
        k_fold, accuracy_threshold
    )

    trainer.run()
    trainer.plot()


def train(path              : str,
          criterion         : _Loss | _WeightedLoss= nn.MSELoss,
          batch_size        : int                  = 10,
          k_fold            : int                  = 5,
          num_epochs        : int                  = 50,
          num_hidden_input  : int                  = 5,
          num_hidden_output : int                  = 3,
          hidden_input_size : int                  = 50,
          hidden_output_size: int                  = 30,
          accuracy_threshold: float                = 0.94) -> None:
    r"""
    Run the training. The input `path` can be either a 
    path to a CSV file that contains the dataset, or to
    a folder with multiple CSV files (so multiple datasets).
    In the first case the training procedure will be runned
    only for that file, otherwise, in the second case,
    multiple times as the number of files. 

    :param path: the path to a CSV file or CSV folder
    :param criterion: the PyTorch type of loss (or a custom one)
    :param batch_size: the Size of the batch for the dataloader
    :param k_fold: number of cross fold validation
    :param num_epochs: The total number of epochs
    :param num_hidden_input: The number of hidden layer in input side
    :param num_hidden_output: The number of hidden layer in output side
    :param hidden_input_size: The number of neurons for each input hidden layer
    :param hidden_output_size: The number of neurons for each output hidden layer
    :param accuracy_threshold: Stop when the current accuracy overcome a value
    :return:
    """
    # Check if the input path is a file or a folder
    input_abspath = opath.abspath(path)
    assert opath.exists(input_abspath), \
        f"[!!!] ERROR: The input path: {path} does not exists."
    
    # If it is a file then create a FSMLOneMeanStdDataset 
    # and run the training
    if opath.isfile(input_abspath):
        train_dataset = FSMLOneMeanStdDataset(input_abspath)
        return __train_one(
            train_dataset, criterion,
            batch_size, k_fold, num_epochs,
            num_hidden_input, num_hidden_output,
            hidden_input_size, hidden_output_size,
            accuracy_threshold
        )
    
    # Otherwise it is a folder
    print(f"[*] Received in input a path: {path}")
    train_multi_dataset = FSMLMeanStdDataset(input_abspath)
    print(f"[*] Created the Multi-Dataset")
    print(train_multi_dataset)

    for idx, train_dataset in enumerate(train_multi_dataset):
        print(f": ---------------- : ({idx}) Using {train_dataset.csv_file} : ---------------- :")
        __train_one(
            train_dataset, criterion,
            batch_size, k_fold, num_epochs,
            num_hidden_input, num_hidden_output,
            hidden_input_size, hidden_output_size,
            accuracy_threshold
        )

    return


def main() -> None:
    file = os.path.join(os.getcwd(), "data/meanstd/")
    train(file, k_fold=3, num_epochs=150)


if __name__ == "__main__":
    main()