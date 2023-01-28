import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import List
from dataclasses import dataclass


@dataclass(frozen=True)
class FSMLNetworkConfiguration:
    r"""
    A class that must be used as input to the :class:`FSMLNet` class
    as a sort of NN configuration, with the total number of layers
    and the dimension of input and output layers (num of neurons).

    Attributes
    ----------
    numLayersInputSide : int
        Number of layers on input side
    inputLayerSize : List[int]
        The output size of each input layer (except for the first one)
    numLayersOutputSide : int
        Number of layers on the output side
    outputLayersSize : List[int]
        The output size of each output layer (except for the last one)
    useDropOut : bool
        True to use also drop out layers
    dropOutRate : float
        The dropout rate to use if dropout is enabled
    """
    numLayersInputSide  : int       # Number of layers on input side
    inputLayersSize     : List[int] # The output size of each input layer (except for the first one)
    numLayersOutputSide : int       # Number of layers on the output side
    outputLayersSize    : List[int] # The output size of each output layer (except for the last one)


class FSMLSimpleNetwork(nn.Module):
    def __init__(self, input_size: int) -> None:
        """
        :param input_size: the input size of the first layer
        """
        super(FSMLSimpleNetwork, self).__init__()

        self.layer1 = nn.Linear(input_size, 50)
        self.layer2 = nn.Linear(50, 40)
        self.layer3 = nn.Linear(40, 20)
        self.layer4 = nn.Linear(20, 10)
        self.layer5 = nn.Linear(10, 5)
        self.layer6 = nn.Linear(5, 2)
    
    def forward(self, data):
        outputs = []

        for single_data in data:
            h = F.relu(self.layer1(single_data))
            h = F.relu(self.layer2(h))
            h = F.relu(self.layer3(h))
            h = F.relu(self.layer4(h))
            h = F.relu(self.layer5(h))
            h = self.layer6(h)
            
            outputs.append(h)
        
        return outputs


class FSMLNet(nn.Module):
    r"""  """
    def __init__(self, network_configuration: FSMLNetworkConfiguration) -> None:
        """
        :param network_configuration: a configuration for the NN
        """
        super(FSMLNet, self).__init__()

        numInputs = 1 # Dimension of the input
        self.inputSideLayers = []
        for i in range(network_configuration.numLayersInputSide):
            layer = nn.Linear(numInputs, network_configuration.inputLayersSize)
            self.inputSideLayers.append(layer)
            self.add_module(f"iLayer{i}", layer)
            numInputs = network_configuration.inputLayersSize
        
        