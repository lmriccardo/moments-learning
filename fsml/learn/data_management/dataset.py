import torch
import torch.nn as nn
from torch.utils.data import Dataset

from typing import Tuple, Iterable, List, Generator, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import fsml.utils as utils
from copy import deepcopy
import os.path as opath


class FSMLOneMeanStdDataset(Dataset):
    r"""
    This class represent a :class:`FSMLOneMeanStdDataset`. 

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
    csv_file : str
        the input path of the CSV file with the data
    input_data : List[List[float]]
        A list with all the input data (parameters)
    input_size : int
        The size of the input data (i.e. the number points in a single tensor)
    output_data : List[List[float]]
        A list with all the output data (mean and std for each specie)
    output_size : int
        The size of the output data (i.e. the number points in a single tensor)
    num_data : int
        The total number of samples
    parameters : List[str]
        A list with all parameters name
    outputs : List[str]
        A list with all the output names
    train_data : Tuple[List[List[float]], List[float]]
        A list with all the train data
    test_data : Tuple[List[List[float]], List[float]]
        A list with all the test data
    """
    def __init__(self, csv_file: str, train: float=0.8, test: float=0.2) -> None:
        """
        :param csv_file: the input path of the CSV file with the data
        """
        super(FSMLOneMeanStdDataset, self).__init__()
        self.csv_file = csv_file

        self.input_data  = [] # A list with all the input data (parameters)
        self.input_size  = 0  # The size of the input data (i.e. the number points in a single tensor)
        self.output_data = [] # A list with all the output data (mean and std for each specie)
        self.output_size = 0  # The size of the output data (i.e. the number points in a single tensor)
        self.num_data    = 0  # The total number of samples 
        self.parameters  = [] # A list with all parameters name
        self.outputs     = [] # A list with all the output names
        self.train_data  = [] # A list with all the train data
        self.test_data   = [] # A list with all the test data
        self.train_size  = 0  # The size of the train set
        self.test_size   = 0  # The size of the test set

        self.is_train = True

        self.__initialize_all(train, test) # Init all the fields

    def test(self) -> None:
        """ Set the dataset for testing """
        self.is_train = False
 
    @staticmethod
    def __split(columns: Iterable[str]) -> Tuple[List[str], List[str]]:
        """ Split the input columns to find the parameters and the variables names """
        # I have structured the data such that all the output variables
        # are written in uppercase, and the parameters in lowercase.
        parameters = [col for col in columns if col.islower() and col != "time"]
        outputs    = [col for col in columns if col.isupper()]
        return parameters, outputs

    def __initialize_all(self, train: float, test: float) -> None:
        """ Initialize all the fields of the class """
        # First get the total number of data and the data itself
        self.num_data, points = utils.read_csv_content(self.csv_file)

        # Then we need to split the data taking the inputs and the output colums
        params, output    = FSMLOneMeanStdDataset.__split(points)
        self.input_data  += points.loc[:, params].values.tolist()
        self.output_data += points.loc[:, output].values.tolist()
        self.input_size   = len(params)
        self.output_size  = len(output)
        self.parameters   = params
        self.outputs      = output

        # Define the standard scaler to preprocess data
        scaler = StandardScaler()
        scaler.fit(self.input_data)
        self.input_data = scaler.transform(self.input_data)

        train_x_data, test_x_data, train_y_data, test_y_data = train_test_split(
            self.input_data, self.output_data, train_size=train, test_size=test, random_state=42
        )

        self.train_data = (train_x_data, train_y_data)
        self.test_data  = (test_x_data,  test_y_data)

        self.train_size = len(self.train_data[0])
        self.test_size  = len(self.test_data[0])
    
    def __len__(self) -> int:
        """ Return the total number of sample in the dataset """
        return self.train_size if self.is_train else self.test_size

    def __repr__(self) -> str:
        filepath_linux_format = opath.basename(self.csv_file).replace('\\', '/')
        csv_filename = opath.basename(filepath_linux_format)

        return f"{self.__class__.__name__}(\n"         + \
               f"   csv_file={csv_filename},\n"        + \
               f"   num_samples={self.num_data},\n"    + \
               f"   input_size={self.input_size},\n"   + \
               f"   output_size={self.output_size},\n" + \
               f"   train_size={self.train_size},\n"   + \
               f"   test_size={self.test_size}\n)"
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Return the i-th data as a tuple of (parameters, means and std.) for train or test """
        # Set the right data set to be used
        current_data = self.train_data
        current_size = self.train_size
        if not self.is_train:
            current_data = self.test_data
            current_size = self.test_size

        # First check that the index is in range
        assert index < current_size, \
            f"[!!!] ERROR: Trying to address element at position {index} " + \
            f"but only {current_size} are available in the dataset"
        
        input_tensor  = torch.tensor(current_data[0][index], dtype=torch.float32)
        output_tensor = torch.tensor(current_data[1][index], dtype=torch.float32)

        return input_tensor, output_tensor, self.output_size
    
    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """ Iterate all the dataset and generate one sample at a time """
        for sample_id in range(self.num_data):
            yield self.__getitem__(sample_id)


class FSMLMeanStdDataset(Dataset):
    r""" A class for the :class:`FSMLMeanStdDataset`
    
    This dataset represent a mix of different dataset, where each
    of the single dataset is given by a CSV file contained in the
    input `data_path`. This dataset can be used to train a number
    of different models, one per dataset, or a single model on the
    entire dataset, where the entire dataset merge all the N
    dataset contained. This mixing is given by a parameter called
    `mixup`. In the case `mixup` is true then the method `__getitem__`
    will return a mixture of samples. 

    Attributes
    ----------
    files : List[str]
        A list with all the files
    num_files : int
        The total number of files in the folder
    count_per_file : Dict[str, int]
        Count the total number of elements for that file
    train_datasets : List[:class:`FSMLOneMeanStdDataset`]
        A list with all the dataset for training
    test_datasets : List[:class:`FSMLOneMeanStdDataset`]
        A list with all the dataset for testing
    max_parameters : int
        The number of maximum parameters overall (from the different dataset)
    max_outputs : int
        The number of maximum output shape overall (from the different dataset)
    total_data : int
        The total amount of data mixing all the datasets
    train_size : int 
        The total amount of training data
    test_size : int
        The total amount of test data
    is_train : bool
        True then use the training data, False use test data

    Methods
    -------
    test(self) -> None
        Set the dataset for testing instead of training
    """
    def __init__(self, data_path: str, mixup: bool=False, train: float=0.8, test: float=0.2) -> None:
        """
        :param data_path: the absolute path with all the datasets (CSV files)
        :param train: the percentage of element that go in the train set
        :param test: the percentage of element that go in the test set
        :param mixup: False means that __getitem__ will returns the i-th dataset,
                      True means that __getitem__ will returns the i-th element overall the 
        """
        super(FSMLMeanStdDataset, self).__init__()
        self.data_path = data_path
        self.mixup     = mixup

        self.files          = []     # A list with all the files
        self.num_files      = 0      # The total number of files in the folder
        self.count_per_file = dict() # Count the total number of elements for that file
        self.total_data     = 0      # The total amount of data mixing all the datasets
        self.max_parameters = 0      # The number of maximum parameters overall (from the different dataset)
        self.max_outputs    = 0      # The number of maximum output shape overall (from the different dataset)
        self.datasets       = []     # A list with all the dataset

        self.train_datasets       = [] # A list with all the dataset for training
        self.train_size           = 0  # The amount of data for training

        self.test_datasets       = [] # A list with all the dataset for testing
        self.test_size           = 0  # The amount of data for testing

        self.is_train = True   # True if __getitem__ should returns train data or False for test data
        
        self.__initialize_all(train, test) # Initialize all the fields
    
    def __initialize_all(self, train: float, test: float) -> None:
        """ Initialize all the fields of the class """
        # First let's count the total number of files in the data folder
        count_condition = lambda x: opath.isfile(x) and x.endswith(".csv")
        self.num_files, self.files = utils.count_folder_elements(self.data_path, count_condition)

        datasets_list: List[FSMLOneMeanStdDataset] = []
        for file in self.files:
            # Create the dataset with the current file of the iteration
            current_dataset = FSMLOneMeanStdDataset(file)
            self.count_per_file[file] = current_dataset.num_data
            self.total_data += current_dataset.num_data

            # We need to update maximum parameters and output numbers
            if current_dataset.input_size > self.max_parameters:
                self.max_parameters = current_dataset.input_size
            
            if current_dataset.output_size > self.max_outputs:
                self.max_outputs = current_dataset.output_size

            # Add the dataset to the list
            datasets_list.append(current_dataset)

        self.datasets = datasets_list

        # Now split the dataset into train and test
        train_data, test_data = train_test_split(datasets_list, 
                                                 train_size=train, 
                                                 test_size=test, 
                                                 random_state=42
        )

        self.train_size, self.test_size = len(train_data), len(test_data)
        self.train_datasets = train_data
        self.test_datasets = test_data

    def test(self) -> None:
        """ Set the dataset for testing instead of training """
        self.is_train = False

    def __len__(self) -> int:
        """ Return the total number of elements of the dataset """
        # If mixup is false then returns the number of datasets
        if not self.mixup:
            return self.num_files

        # Otherwise we need to count the number of elements
        # for each dataset in the dataset list
        return self.total_data
    
    def __repr__(self) -> str:
        """ Return a string representation of the dataset """
        if not self.mixup:
            return f"{self.__class__.__name__}(\n"          + \
                   f"   num_total_data={self.num_files},\n" + \
                   f"   num_files={self.num_files},\n"      + \
                   f"   data_folder={self.data_path}\n)"
    
        return f"{self.__class__.__name__}(\n"               + \
               f"   num_total_data={self.total_data},\n"     + \
               f"   max_parameters={self.max_parameters},\n" + \
               f"   max_outputs={self.max_outputs},\n"       + \
               f"   data_path={self.data_path},\n"           + \
               f"   train_size={self.train_size},\n"         + \
               f"   test_size={self.test_size}\n)"
    
    def __getitem_not_mixup(self, index) -> FSMLOneMeanStdDataset:
        """ __getitem__ implementation when mixup is False """
        assert index < self.num_files, \
            f"[!!!] ERROR: Please select an index less then {self.num_files}"
        
        return self.datasets[index]
    
    def __getitem_mixup(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ __getitem__ implementation when mixup is True """
        # Select train datasets or test datasets
        current_dataset = self.train_datasets
        current_ds_size = self.train_size
        if not self.is_train:
            current_dataset = self.test_datasets
            current_ds_size = self.test_size
        
        # Check that the index is not out of bound
        assert index < current_ds_size, \
            f"[!!!] ERROR: Please select an index less than {current_ds_size}"
        
        # Then take the element from the corresponding dataset
        current_value = 0
        final_dataset_item = None
        for ds in current_dataset:
            if current_value + ds.num_data > index:
                final_dataset_item = ds[index - (current_value + ds.num_data)]
                break
            
            current_value += ds.num_data
        
        # Pad if necessarily
        input_data, output_data = final_dataset_item
        input_data = nn.functional.pad(input_data, pad=(0, self.max_parameters - input_data.shape[0]))
        output_data = nn.functional.pad(output_data, pad=(0, self.max_outputs - output_data.shape[0]))

        return input_data, output_data
    
    def __getitem__(self, index: int) -> Union[FSMLOneMeanStdDataset, Tuple[torch.Tensor, torch.Tensor]]:
        """ Get the i-th item """
        if not self.mixup:
            return self.__getitem_not_mixup(index)

        return self.__getitem_mixup(index)
    
    def __iter__(self) -> Generator[FSMLOneMeanStdDataset, None, None]:
        for index in range(self.__len__()):
            yield self.__getitem__(index)
    

def get_dataset_by_indices(
    src_dataset: FSMLOneMeanStdDataset, train_ids: List[int], test_ids: List[int]
) -> FSMLOneMeanStdDataset:
    r"""
    From an input already existing :class:`FSMLOneMeanStdDataset` create
    a new dataset of the same type, by copying it, but selecting
    only a portion of the train and test set. This portion is identified
    by the input train indexes and test indexes.

    :param src_dataset: The input already existing dataset
    :param train_ids: The indexes for the new train set
    :param test_ids: The indexes for the new test set
    :return: a new dataset with "filtered" train and test set
    """
    # First copy the old dataset into the new one
    dataset = deepcopy(src_dataset)

    # Now split the train data both input and output
    train_x_data, train_y_data = dataset.train_data

    # Select only a subset of train data
    train_x = torch.tensor(train_x_data, dtype=torch.float32)
    train_X = train_x[train_ids, :].tolist()
    train_y = torch.tensor(train_y_data, dtype=torch.float32)
    train_Y = train_y[train_ids, :].tolist()

    # Then do the same thing but with test indexes
    test_X = train_x[test_ids, :].tolist()
    test_Y = train_y[test_ids, :].tolist()

    dataset.train_data = (train_X, train_Y)
    dataset.train_size = len(train_X)
    dataset.test_data  = (test_X, test_Y)
    dataset.test_size  = len(test_X)

    dataset.num_data = dataset.train_size + dataset.test_size
    dataset.is_train = True

    return dataset