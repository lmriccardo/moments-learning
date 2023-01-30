import torch
import torch.nn as nn
from torch.utils.data import Dataset

from typing import Tuple, Iterable, List, Generator
import fsml.utils as utils
import os.path as opath
import pandas as pd


class FSMLMeanStdDataset(Dataset):
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
        super(FSMLMeanStdDataset, self).__init__()

        self.data_path      = data_path  # The input folder with all the data
        self.num_samples    = 0          # Total number of simulations
        self.count_per_file = dict()     # Count the total number of actual simulations per file
        self.num_files      = 0          # Total number of files
        self.files          = []         # A list with all the files
        self.samples        = []         # A list that contains all the sample from all the files
        self.max_parameters = 0          # The maximum number of parameters in the dataset
        
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
            params, outputs = FSMLMeanStdDataset.__split(current_csv_content.columns)

            # Compute the length of the parameters and update the maximum parameters
            if (num_params := len(params)) > self.max_parameters:
                self.max_parameters = num_params

            self.samples_param_output_dict.update({ sid : (params, outputs) for sid in sample_range })

            self.count_per_file[file] = current_number_perfile
            total_number_of_sample += current_number_perfile
        
        self.num_samples = total_number_of_sample

    def __len__(self) -> int:
        """ Return the length of the dataset """
        return self.num_samples

    def __repr__(self) -> str:
        """ Return a string representation of the dataset """
        return f"{self.__class__.__name__}(\n"                 + \
               f"  data_folder={self.data_path},\n"            + \
               f"  num_files={self.num_files},\n"              + \
               f"  total_num_samples={self.num_samples},\n"    + \
               f"  maximum_parameters={self.max_parameters}\n" + \
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

        # Create tensors for input data and output data
        tensor_input_data = torch.tensor(input_data)
        tensor_output_data = torch.tensor(output_data)

        # zero pad the input tensor for create a fixed length sample
        pad_number = self.max_parameters - params_len
        tensor_input_data = nn.functional.pad(tensor_input_data, pad=(0, pad_number))

        return tensor_input_data, tensor_output_data

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """ Iterate all the dataset and generate one sample at a time """
        for sample_id in range(self.num_samples):
            yield self.__getitem__(sample_id)


class FSMLOneStepAheadDataset(Dataset):
    """ """
    def __init__(self, data_path: str) -> None:
        """
        :param data_path: The fully qualified path to the data folder
                          with all the dense output of the simulations.
                          Inside the folder there should be all the file
                          required for the dataset and the training.
        """
        super(FSMLOneStepAheadDataset, self).__init__()

        self.data_path       = data_path  # The absolute path of the data folder with all the CSV files
        self.input_data      = []         # The input data to the Neural Network
        self.output_data     = []         # The output data, i.e., the ground trouth
        self.files           = []         # A list with all the files in the data folder
        self.num_files       = 0          # The total number of files
        self.num_samples     = 0          # The total number of simulations
        self.max_input_size  = 0          # the maximum lenght of the input data
        self.max_output_size = 0          # The maximum lenght of the output data

        self.__initialize_all()

    @staticmethod
    def __split(columns: Iterable[str]) -> Tuple[List[str], List[str]]:
        """ Split the input columns to find the parameters and the variables names """
        # I have structured the data such that all the output variables
        # are written in uppercase, and the parameters in lowercase.
        parameters = [col for col in columns if col.islower() and col != "time"]
        outputs    = [col for col in columns if col.endswith("_amount")]
        return parameters, outputs

    def __initialize_all(self) -> None:
        """ Initialize all the fields of the class """
        # First let's count the total number of files in the data folder
        count_condition = lambda x: opath.isfile(x) and x.endswith(".csv")
        self.num_files, self.files = utils.count_folder_elements(self.data_path, count_condition)

        for file in self.files:
            _, current_csv_content = utils.count_csv_rows(file)

            # First we need to find where each simulation of the current
            # model ends, essentially when the time columns is reset to 0
            params, outputs = FSMLOneStepAheadDataset.__split(current_csv_content.columns)
            
            # Update the maximum size of features with which we will pad
            if (tot_len := len(params) + len(outputs)) > self.max_input_size:
                self.max_input_size = tot_len

            # Update the maximum size of the output data
            if (tot_output := len(outputs)) > self.max_output_size:
                self.max_output_size = tot_output

            variables = params + outputs

            # Obtain the indexes where each time the time column is reset to 0
            simulation_indexes = [0] + utils.find_indexes(current_csv_content)[1:]
            simulation_indexes += [simulation_indexes[-1] + simulation_indexes[1]]
            content_no_time = current_csv_content.loc[:, variables]
            for start_index, end_index in zip(simulation_indexes[:-1], simulation_indexes[1:]):
                self.input_data += content_no_time.iloc[start_index:end_index - 1, :].values.tolist()
                self.output_data += content_no_time.loc[:, outputs].iloc[start_index + 1:end_index, :].values.tolist()
        
        # Check that at the end input and output data have the same length
        assert len(self.input_data) == len(self.output_data), \
            "[!!!] ERROR: Input data List and Output Data list of different size: " + \
           f"Input data Size {len(self.input_data)} - {len(self.output_data)} Output Data Size"

        self.num_samples = len(self.input_data)

    def __len__(self) -> int:
        """ Return the length of the dataset """
        return self.num_samples

    def __repr__(self) -> str:
        """ Return a string representation of the dataset """
        return f"{self.__class__.__name__}(\n"                 + \
               f"  data_folder={self.data_path},\n"            + \
               f"  num_files={self.num_files},\n"              + \
               f"  total_num_samples={self.num_samples},\n"    + \
               f"  maximum_header_length={self.max_number}\n" + \
               ")"
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """ Return the i-th data as a tuple of (parameters, means and std.) """
        # First check that the index is in range
        assert index < self.num_samples, \
            f"[!!!] ERROR: Trying to address element at position {index} " + \
            f"but only {self.num_samples} are available in the dataset"
        
        # First take the data (input and output)
        x_data = torch.tensor(self.input_data[index])
        y_data = torch.tensor(self.output_data[index])

        # Then pad the input data with zeros
        num_pad = self.max_input_size - x_data.shape[0]
        x_data = nn.functional.pad(x_data, pad=(0, num_pad))

        # Pad also the output
        current_output_shape = y_data.shape[0]
        num_pad = self.max_output_size - y_data.shape[0]
        y_data = nn.functional.pad(y_data, pad=(0, num_pad))

        return x_data, y_data, current_output_shape
    
    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """ Iterate all the dataset and generate one sample at a time """
        for sample_id in range(self.num_samples):
            yield self.__getitem__(sample_id)