import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss

import fsml.utils as utils
from fsml.learn.data_mangement.dataset import FSMLOneMeanStdDataset
from fsml.learn.data_mangement.dataloader import FSMLDataLoader
from sklearn.model_selection import KFold
import fsml.learn.models.nets as nets
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm
import os
import time


class MagnitudeLoss(nn.Module):
    def __init__(self) -> None:
        super(MagnitudeLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, indexes: List[int]) -> torch.Tensor:
        torch_indexes = torch.tensor(indexes)
        total_loss = torch.zeros(1)
        for idx, (row1, row2) in enumerate(zip(inputs, targets)):
            row1, row2 = row1[:indexes[idx]], row2[:indexes[idx]]

            # Compute the differences between order of magnitudes
            rate_order_of_magnitudes = torch.abs(torch.log10(abs(row1 / row2)))
            
            # Then takes order of magnitude away so that we can just compare the raw numbers
            of_magnitude_row1 = torch.log10(row1.abs()).type(torch.int32)
            of_magnitude_row2 = torch.log10(row2.abs()).type(torch.int32)
            row1 = row1 / torch.tensor([10.0]).pow(of_magnitude_row1)
            row2 = row2 / torch.tensor([10.0]).pow(of_magnitude_row2)

            # Now compute the losses between the raw values
            raw_loss = torch.mean((row1 - row2).pow(2))

            # Compute the mean of all the order of magnitudes
            mean_rate_of_magnitude = torch.mean(rate_order_of_magnitudes)

            total_loss += raw_loss.pow(mean_rate_of_magnitude)

        return total_loss / torch_indexes.shape[0]
    

class KFoldCrossValidationWrapper:
    r""" A simple wrapper class for K-Fold Cross Validation """

    @staticmethod
    def setup_kFold_validation(dataset    : FSMLOneMeanStdDataset, 
                               kf_split   : int, 
                               batch_size : int) -> List[Tuple[int, FSMLDataLoader]]:
        """
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
            print(fold_num, train_ids, test_ids)
    

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

    # def _train_step()
    

def train_one(csv_file: str, num_epochs: int=10) -> None:
    r"""
    Execute train and test for one dataset given by the input CSV file

    :param csv_file: the fully qualified path to the CSV file
    :param num_epochs: The number of epochs to run
    :return:
    """
    print(f"[*] Input CSV Dataset file: {csv_file}")

    # First create the dataset and then the dataloader
    print("[*] Creating the respective dataset")
    csv_ds = FSMLOneMeanStdDataset(csv_file)
    print(csv_ds)

    print("[*] Creating the dataloader")
    csv_dl = FSMLDataLoader(csv_ds, 10, shuffle=True, drop_last=True)

    # Then instanciate the model
    print("[*] Creating the predictor model")
    fsml_predictor = nets.FSML_MLP_Predictor(
        csv_ds.input_size,  5, 50,
        csv_ds.output_size, 3, 30
    )
    print(fsml_predictor)
    
    # Instanciate the optimizer and the loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(fsml_predictor.parameters(), lr=0.0001)

    fsml_predictor.train()
    train_losses = []
    train_accs = []
    torch.random.manual_seed(42)

    print("[*] Starting training the model")
    start = time.time()
    for epoch in range(num_epochs):

        train_loss = torch.zeros(1)
        train_acc  = 0
        train_step = 0
        progress_bar = tqdm(csv_dl(), desc=f"Epoch: {epoch} --- ", leave=True)
        optimizer.zero_grad()
        for x_data, y_data, _ in progress_bar:

            output      = fsml_predictor(x_data)
            loss        = criterion(output, y_data)
            train_loss += loss
            train_step += 1

            with torch.no_grad():
                acc = utils.compute_accuracy(output, y_data)
                train_acc += acc
            
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix_str(
                f"Train Loss: {(train_loss / train_step).item()}" + \
                f"- Acc: {train_acc / train_step}")
            
            progress_bar.refresh()
        
        train_loss /= train_step
        train_acc /= train_step
        train_losses.append(train_loss.item())
        train_accs.append(train_acc)
    
    end = time.time()
    print(f"[*] Training procedure ended in {end - start} sec")

    epochs = list(range(0, num_epochs))
    plt.plot(epochs, train_losses, label="Train losses over epochs")
    plt.plot(epochs, train_accs, label="Train Accuracy over epochs")
    plt.xlabel("Number Of Epochs")
    plt.ylabel("Train losses and Accuracy")
    plt.legend(loc="upper right")
    plt.show()


def train() -> None:
    file = os.path.join(os.getcwd(), "data/meanstd/BIOMD00002_MeanStd.csv")
    train_one(file, num_epochs=150)


def kfold_try() -> None:
    file = os.path.join(os.getcwd(), "data/meanstd/BIOMD00002_MeanStd.csv")
    csv_ds = FSMLOneMeanStdDataset(file)

    KFoldCrossValidationWrapper.setup_kFold_validation(csv_ds, 5, 32)


if __name__ == "__main__":
    # train()
    kfold_try()