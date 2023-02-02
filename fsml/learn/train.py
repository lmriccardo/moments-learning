import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import fsml.learn.data_mangement.dataset as dataset
import fsml.learn.data_mangement.dataloader as dataloader
import fsml.learn.models.nets as nets
import fsml.utils as utils
import matplotlib.pyplot as plt
from typing import List
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
    csv_ds = dataset.FSMLOneMeanStdDataset(csv_file)
    print(csv_ds)

    print("[*] Creating the dataloader")
    csv_dl = dataloader.FSMLDataLoader(csv_ds, 10, shuffle=True, drop_last=True)

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


if __name__ == "__main__":
    train()