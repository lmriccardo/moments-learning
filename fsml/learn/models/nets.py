import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FSML_MLP_Predictor(nn.Module):
    def __init__(self, input_dimension    : int,
                       input_side_layers  : int,
                       input_hidden_size  : int,
                       output_dimension   : int,
                       output_side_layers : int,
                       output_hidden_size : int) -> None:
        """
        :param input_dimension: the dimensionality of the input
        :param input_side_layers: how many layers in the input side
        :param input_hidden_size: the dimensionality of the hidden input side layers
        :param output_dimension: the dimensionality of the output
        :param output_side_layers: how many layers on output side
        :param output_hidden_size: the dimensionality of the hidden output side layers
        """
        super(FSML_MLP_Predictor, self).__init__()

        # The final list with all the layers
        self.layers = []

        # First add the input side layers
        current_in_dimension = input_dimension
        for iside_layer in range(input_side_layers):
            current_layer = nn.Linear(current_in_dimension, input_hidden_size)
            self.add_module(f"iLayer{iside_layer}", current_layer)
            self.layers.append(current_layer)
            current_in_dimension = input_hidden_size
        
        # Then add output side layers
        current_out_dimension = output_hidden_size
        for oside_layer in range(output_side_layers):
            if oside_layer == output_side_layers - 1:
                current_out_dimension = output_dimension

            current_layer = nn.Linear(current_in_dimension, current_out_dimension)
            self.add_module(f"oLayer{oside_layer}", current_layer)
            self.layers.append(current_layer)
            current_in_dimension = output_hidden_size
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """ The forward function for the neural network """
        h = Variable(data)
        for idx, layer in enumerate(self.layers):
            h = layer(h)

            # Apply the relu only if it is not the output
            if idx != len(self.layers) - 1:
                h = F.relu(h)
        
        return h