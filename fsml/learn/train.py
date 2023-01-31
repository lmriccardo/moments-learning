import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import fsml.learn.data_mangement.dataset as dataset
import fsml.learn.data_mangement.dataloader as dataloader
import fsml.learn.models.nets as nets
from typing import List
from tqdm import tqdm
import os.path as opath
import os 


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


def iter_dataset():
    # data_dir = opath.join(os.getcwd(), "data/meanstd/")
    # ds = dataset.FSMLMeanStdDataset(data_dir)
    # dl = dataloader.FSMLDataLoader(ds, 3, True)
    # model = nets.FSMLSimpleNetwork(input_size=ds.max_parameters)
    # for data in dl():
    #     input_data, _ = data
    #     output = model(input_data)
    #     print(output)

    data_dir = opath.join(os.getcwd(), "data/denseoutputs/")
    denseoutput_ds = dataset.FSMLOneStepAheadDataset(data_dir)
    denseoutput_dl = dataloader.FSMLDataLoader(denseoutput_ds, 32, True, drop_last=True)
    osa_predictor = nets.FSML_OSA_Predictor(
        denseoutput_ds.max_input_size,  5, 50,
        denseoutput_ds.max_output_size, 5, 30
    )
    
    loss = nn.MSELoss()
    epochs = 10
    # optimizer = optim.Adam(osa_predictor.parameters(), lr=0.00001, weight_decay=1e-6)
    optimizer = optim.Adam(osa_predictor.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2,  patience=35, verbose=True, min_lr=1e-06)
    train_losses = []
    osa_predictor.train()

    torch.random.manual_seed(42)

    for epoch in range(epochs):
        train_loss = torch.zeros(1)
        train_step = 0
        progress_bar = tqdm(denseoutput_dl(), desc=f"Epoch Number: {epoch} --- ", leave=True)
        for x_data, y_data, _ in progress_bar:
            output = osa_predictor(x_data)
            total_loss = loss(output, y_data)
            train_loss += total_loss
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(osa_predictor.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()
            train_step += 1
            progress_bar.set_postfix_str(f"Train Loss in Log Scale: {(train_loss / train_step).item()}")
            progress_bar.refresh()
        
        train_loss /= train_step
        # scheduler.step(train_loss)
        train_losses.append(train_loss.item())


if __name__ == "__main__":
    iter_dataset()