import torch
import torch.nn as nn

from typing import Callable, Dict


class RBFActivationFunctions:

    @staticmethod
    def gaussian(alpha: torch.Tensor) -> torch.Tensor:
        r""" Gaussian Radial Basis Function """
        return torch.exp(-1 * alpha.pow(2))

    @staticmethod
    def quadratic(alpha: torch.Tensor) -> torch.Tensor:
        r""" Quadratic Radial Basis Function """
        return alpha.pow(2)

    @staticmethod
    def inverse_quadratic(alpha: torch.Tensor) -> torch.Tensor:
        r""" Inverse Quadratic Radial Basis Function """
        return torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))

    @staticmethod
    def multiquadric(alpha: torch.Tensor) -> torch.Tensor:
        r""" Multiquadratic Radial Basis Function """
        return (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)

    @staticmethod
    def inverse_multiquadric(alpha: torch.Tensor) -> torch.Tensor:
        r""" Inverse Multiquadratic Radial Basis Function """
        return torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)

    @staticmethod
    def spline(alpha: torch.Tensor) -> torch.Tensor:
        r""" Spline Radial Basis Function """
        return (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))

    @staticmethod
    def poisson(alpha: torch.Tensor) -> torch.Tensor:
        r""" Poisson Radial Basis Function """
        return (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    
    def functions() -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        return {
            "gaussian"               : RBFActivationFunctions.gaussian,
            "quadratic"              : RBFActivationFunctions.quadratic,
            "inverse quadratic"      : RBFActivationFunctions.inverse_quadratic,
            "multiquadratic"         : RBFActivationFunctions.multiquadric,
            "inverse multiquadratic" : RBFActivationFunctions.inverse_multiquadric,
            "spline"                 : RBFActivationFunctions.spline,
            "poisson"                : RBFActivationFunctions.poisson
        }

    @staticmethod
    def get_function(func_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        return RBFActivationFunctions.functions[func_name.lower()]
    

class RBF_Layer(nn.Module):
    r"""
    Transforms incoming data using a given radial basis function:

    .. math:
        u_{i} = \text{rbf}(\frac{\|x - c_{i}\|}{s_{i}}).
    
    Attributes
    ----------
    centres : nn.Parameter
        the learnable centres of shape (out_features, in_features).
        The values are initialised from a standard normal distribution.
        Normalising inputs to have mean 0 and standard deviation 1 is
        recommended.
    
    log_sigmas : nn.Parameter
        logarithm of the learnable scaling factors of shape (out_features).
    
    basis_func : Callable[[torch.Tensor], torch.Tensor]
        the radial basis function used to transform the scaled distances.
    """
    def __init__(self, in_features : int, 
                       out_features: int, 
                       basis_func  : Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        :param in_features: the number of features of the input data
        :param out_features: the number of features to return
        :param basic_func: The basis function to be applied
        """
        super(RBF_Layer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.basis_function = basis_func

        self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.ParameterDict(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Reset the parameters centers and log_sigmas """
        nn.init.normal_(self.centers, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, x_data: torch.Tensor) -> torch.Tensor:
        size = (x_data.size(0), self.out_features, self.in_features)
        x = x_data.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_function(distances)
    

class FSML_RBF_Predictor(nn.Module):
    r""" A simple RBF network implementing RBF layers """
    def __init__(self, in_features   : int, 
                       out_features  : int,
                       n_hidden_layer: int,
                       hidden_sizes  : int,
                       basis_func    : Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        :param in_features: the number of features of the input data
        :param out_features: the number of features to return
        :param basic_func: The basis function to be applied
        """
        self.layers = []

        # Add the first RBF and Linear layer
        self.layers.append(RBF_Layer(in_features, hidden_sizes[0], basis_func))
        self.layers.append(nn.Linear(hidden_sizes[0], hidden_sizes[0]))

        for hidden in range(1, n_hidden_layer):
            self.layers.append(RBF_Layer(hidden_sizes[hidden - 1], hidden_sizes[hidden], basis_func))
            self.layers.append(nn.Linear(hidden_sizes[hidden], hidden_sizes[hidden]))

        self.layers.append(RBF_Layer(hidden_sizes[-1], out_features, basis_func))
        self.layers.append(nn.Linear(out_features, out_features))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x_data: torch.Tensor) -> torch.Tensor:
        output = x_data
        for layer in self.layers:
            output = layer(output)

        return output