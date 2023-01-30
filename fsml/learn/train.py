import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import fsml.learn.data_mangement.dataset as dataset
import fsml.learn.data_mangement.dataloader as dataloader
import fsml.learn.models.nets as nets
from tqdm import tqdm
import os.path as opath
import os 


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
    denseoutput_dl = dataloader.FSMLDataLoader(denseoutput_ds, 3, True, drop_last=True)
    osa_predictor = nets.FSML_OSA_Predictor(
        denseoutput_ds.max_input_size,  5, 50,
        denseoutput_ds.max_output_size, 5, 50
    )
    
    loss = nn.MSELoss()
    epochs = 10
    optimizer = optim.Adam(osa_predictor.parameters(), lr=0.0001, weight_decay=1e-5)
    train_losses = []
    osa_predictor.train()

    torch.random.manual_seed(42)
    
    import time 

    for epoch in range(epochs):
        train_loss = 0
        train_step = 0
        for x_data, y_data, output_shapes in denseoutput_dl():
            output = osa_predictor(x_data)
            concat_values = torch.hstack((output, y_data))
            mean_value = torch.mean(concat_values, dim=1, keepdim=True)
            min_value = torch.min(concat_values, dim=1, keepdim=True).values
            max_value = torch.max(concat_values, dim=1, keepdim=True).values
            normalized_output = (output - mean_value) / (max_value - min_value)
            normalized_y_true = (y_data - mean_value) / (max_value - min_value)

            total_loss = torch.zeros(1)
            for oshape, y, y_true in zip(output_shapes, normalized_output, normalized_y_true):
                current_loss = loss(y[:oshape], y_true[:oshape])
                total_loss += current_loss

            train_loss += total_loss.item()
            total_loss.backward()
            optimizer.step()
            train_step += 1
        
        train_loss /= train_step
        train_losses.append(train_loss)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}")


if __name__ == "__main__":
    iter_dataset()