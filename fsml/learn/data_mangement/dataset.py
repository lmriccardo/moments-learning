import torch
from torch.utils.data import Dataset

from typing import Tuple, Iterable, List, Generator
import fsml.utils as utils
import os.path as opath


class FSMLDataset(Dataset):
    r"""
    This class represent a :class:`SimulationDataset`. 

    A class to represent the dataset of all the simulations. Essentially
    the dataset is structured such that, on request (i.e. indexing) or 
    on iteration it returns a couple of elements, where the first element
    represent the input of the neural network, while the second will be
    the ground trouth that will be used to compare the output of the NN
    against it and thus compute the loss and update the optimizer. 

    The dataset is initialized with just the fully qualified path that
    points to the folder containing the result of all the simulations
    that have been performed previously. In that folder there would be
    a bunch of CSV files that contains a single row for each simulation
    of the same model, and a bunch of columns (with a lowercase name)
    representing the parameters of the model, and a bunch of (uppercase)
    columns representing the mean and the standard deviation of species. 

    The on indexing, i.e. call the :meth:`__getitem__` method with an index `i`
    as input the dataset will returns a tuple with just two elements, where
    the first are the input of the neural networks and are composed by the
    parameters of the model, while the second represents the ground trouth
    values is composed by the mean and the standard deviations. 

    Attributes
    ----------
    data_path : str
        The path to the data folder with all the simulations
    num_samples : int 
        Total number of simulations
    count_per_file : Dict[str, int]
        Count the total number of actual simulations per file
    num_files : int
        Total number of files (total number of simulated models)
    files : List[str]
        A list with all the files
    samples : List[float]
        A list that contains all the sample from all the files
    samples_param_output_dict : Dict[int, Tuple[List[str], List[str]]]
        A dictionary that maps for each sample (precisely a sample ID, 
        or essentially the position inside the `samples` list of the sample)
        a couple containing the parameters and species names that will be used
        to split the list of float values into input and output data.
    """
    def __init__(self, data_path: str) -> None:
        """
        :param data_path: the absolute path to the data folder 
                          with the simulation results. Inside the
                          folder there are all the CSV files required.
        """
        super(FSMLDataset, self).__init__()

        self.data_path      = data_path  # The input folder with all the data
        self.num_samples    = 0          # Total number of simulations
        self.count_per_file = dict()     # Count the total number of actual simulations per file
        self.num_files      = 0          # Total number of files
        self.files          = []         # A list with all the files
        self.samples        = []         # A list that contains all the sample from all the files
        
        # Now I define a mapping that for each sample, maps the
        # ID of the sample to the corresponding tuple (input, output)
        # where the inputs are the parameters and the output are the
        # mean and standard deviations of the output species.
        self.samples_param_output_dict = dict()

        self.__initialize_all() # Init all the fields
    
    def __initialize_all(self) -> None:
        """ Initialize all the fields of the class """
        # First let's count the total number of files in the data folder
        count_condition = lambda x: opath.isfile(x) and x.endswith(".csv")
        self.num_files, self.files = utils.count_folder_elements(self.data_path, count_condition)

        # Then let's count the number of simulations for each file
        total_number_of_sample = 0
        for file in self.files:
            current_number_perfile, current_csv_content = utils.count_csv_rows(file)

            # Now we need to post-process the CSV content
            sample_range = range(total_number_of_sample, total_number_of_sample + current_number_perfile)
            self.samples += current_csv_content.iloc[:, 1:].to_numpy().tolist()
            params, outputs = FSMLDataset.__split(current_csv_content.columns)
            self.samples_param_output_dict.update({ sid : (params, outputs) for sid in sample_range })

            self.count_per_file[file] = current_number_perfile
            total_number_of_sample += current_number_perfile
        
        self.num_samples = total_number_of_sample

    def __len__(self) -> int:
        """ Return the length of the dataset """
        return self.num_samples

    def __repr__(self) -> str:
        """ Return a string representation of the dataset """
        return f"{self.__class__.__name__}(\n"             + \
               f"  data_folder={self.data_path},\n"        + \
               f"  num_files={self.num_files},\n"          + \
               f"  total_num_samples={self.num_samples}\n" + \
               ")"
    
    @staticmethod
    def __split(columns: Iterable[str]) -> Tuple[List[str], List[str]]:
        """ Split the input columns to find the parameters and the variables names """
        # I have structured the data such that all the output variables
        # are written in uppercase, and the parameters in lowercase.
        parameters = [col for col in columns if col.islower() and col != "time"]
        outputs    = [col for col in columns if col.isupper()]
        return parameters, outputs
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Return the i-th data as a tuple of (parameters, means and std.) """
        # First check that the index is in range
        assert index < self.num_samples, \
            f"[!!!] ERROR: Trying to address element at position {index} " + \
            f"but only {self.num_samples} are available in the dataset"
        
        # Obtain the parameters list and the output list for that sample
        params, outputs = self.samples_param_output_dict[index]

        # Compute the length of the two list to perform the splitting
        params_len, outputs_len = len(params), len(outputs)

        # Split the data into input data and output data
        current_data = self.samples[index]
        input_data  = current_data[:params_len]
        output_data = current_data[params_len:params_len + outputs_len]

        return torch.tensor(input_data), torch.tensor(output_data)

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """ Iterate all the dataset and generate one sample at a time """
        for sample_id in range(self.num_samples):
            yield self.__getitem__(sample_id)