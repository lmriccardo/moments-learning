import torch
import torch.nn as nn


class DecisionTreeRegressor(nn.Module):
    r""" A simple implementation of a decition tree for regression """
    def __init__(self, depth):
        super(DecisionTreeRegressor, self).__init__()
        self.splits = nn.ModuleList([nn.Linear(1, 1) for _ in range(depth)])
        self.values = nn.ModuleList([nn.Linear(1, 1) for _ in range(depth)])

    def forward(self, x_data: torch.Tensor) -> torch.Tensor:
        for split, value in zip(self.splits, self.values):
            split_value = split(x_data).squeeze(-1)
            x_data = torch.where(x_data < split_value, value(x_data).squeeze(-1), x_data)
        return x_data


class FSML_GBTR_Predictor(nn.Module):
    r""" A simple implementation of Gradient Boosting Tree for regression """
    def __init__(self, n_trees: int, tree_depth: int) -> None:
        super(FSML_GBTR_Predictor, self).__init__()
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.trees = nn.ModuleList([
            
        ])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)