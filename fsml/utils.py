from typing import Tuple, List, Generator, Union, Optional
import basico.biomodels as biomodels
import matplotlib.pyplot as plt
from basico import COPASI
import os.path as opath
import COPASI as co
import pandas as pd
import numpy as np
import libsbml
import random
import math
import sys
import os


# -----------------------------------------------------------------------------
# General Utilities
# -----------------------------------------------------------------------------

def get_ranges(nmax: int, cpu_count: int) -> List[Tuple[int, int]]:
    """
    Given a maximum horizon and the number of cpus return a number
    of ranges between the max number and the cpu count. The number of 
    returned ranges is exactly (nmax / cpu_count) if nmax is a
    multiple of cpu_count, otherwise it is (nmax / cpu_count) + (nmax % cpu_count).

    example:
        >>> get_ranges(67, 16)
        [[0, 16], [16, 32], [32, 48], [48, 64], [64, 67]]
    
    :param nmax:      the maximum horizon
    :param cpu_count: the total number of cpus
    :return: a list containing all the ranges
    """
    count = 0
    ranges = []
    while count < nmax:
        if nmax < cpu_count:
            ranges.append([count, nmax])
        else:
            condition = count + cpu_count < nmax
            operator = cpu_count if condition else nmax % cpu_count
            ranges.append([count, count + operator])

        count += cpu_count

    return ranges


# -----------------------------------------------------------------------------
# Transformation Utilities
# -----------------------------------------------------------------------------

class Errors:
    FILE_NOT_EXISTS  = 1
    XML_READ_ERROR   = 2
    CONVERSION_ERROR = 3


def print_errors(doc: libsbml.SBMLDocument, nerrors: int) -> None:
    """
    Print all the errors for a given SBML document

    :param doc:     A handle to the SBML document
    :param nerrors: The number of errors to be printed
    """
    for i in range(0, nerrors):
        format_string = "[line=%d] (%d) <category: %s, severity: %s> %s"
        error: libsbml.XMLError = doc.getError(i)
        error_line = error.getLine()
        error_id = error.getErrorId()
        error_category = error.getCategory()
        error_severity = error.getSeverity()
        error_message = error.getMessage()

        print(format_string % (
            error_line, error_id, error_category,
            error_severity, error_message
        ))

    exit(Errors.XML_READ_ERROR)


def load_sbml(sbmlpath: str) -> libsbml.SBMLDocument:
    """
    Given an SBML file with the absolute path load the file as an SBML document

    :param sbmlpath: the absolute path to the SBML file
    :return: a handler to the SBMLDocument given by the SBML file
    """
    # Read the SBML and obtain the Document
    reader = libsbml.SBMLReader()
    document = reader.readSBML(sbmlpath)
    
    # Check if there are errors after reading and in case print those errors
    if document.getNumErrors() > 0:
        print_errors(document, document.getNumErrors())

    return document


def save_model(model: libsbml.Model, path: str) -> None:
    """
    Write a SBML Model inside a file

    :param model: a handle to the SBML model
    :param path:  a string representing the output path
    """
    writer = libsbml.SBMLWriter()
    document = model.getSBMLDocument()
    writer.writeSBMLToFile(document, path)


def download_model(prefix_path: str, model_id: int) -> str:
    """
    Download a SBML model given the ID from the BioModels Database

    :param model_id:    the ID of the model that needs to be downloaded
    :param prefix_path: the folder where store the new model
    :return: the path where the model has been stored
    """
    modelname = "BIOMD%05d.xml" % model_id
    sbml_content = biomodels.get_content_for_model(model_id)
    output_file = opath.join(opath.abspath(prefix_path), modelname)
    open_mode = "x" if not opath.exists(output_file) else "w"

    with open(output_file, mode=open_mode, encoding="utf-8") as fhandle:
        fhandle.write(sbml_content)
    
    return output_file


def write_paths(paths: List[str], output_path: str) -> None:
    """
    Save the list of paths inside a file: one path per row

    :param paths:       the list with all the paths
    :param output_path: the file where to store the pats
    :return:
    """
    abs_output_path = opath.abspath(output_path)
    open_mode = "x" if not opath.exists(abs_output_path) else "w"
    with open(abs_output_path, mode=open_mode, encoding="utf-8") as fhandle:
        file_content = "\n".join(paths)
        fhandle.write(file_content)


def remove_original(paths: List[str]) -> None:
    """
    Remove the original model files

    :param paths: the list with all the output paths
    :return:
    """
    for path in paths:
        filename, extension = path.split("_output")
        original_filepath = f"{filename}{extension}"
        os.remove(original_filepath)


def to_integral(params: List[float]) -> List[float]:
    """
    Given a list of parameters convert each parameter from the 
    original value to a value between 1 and 9. That is, given
    a parameter value x we obtain z = x / (10 ** int(log10(x))).

    :param params: the list with all the parameters value
    :return: the new list of parameters
    """
    ceil = lambda x: x + 1 if x > 0 else x - 1
    scale = lambda x: x / (10 ** int(ceil(math.log10(x)) if x != 0.0 else 0.0))
    new_params = list(map(scale, params))
    return new_params


def to_string(vector: Union[np.ndarray, List[float]]) -> str:
    """
    Convert a Numpy vector to a string.

    :param vector: the input vector
    :return: the string representation
    """
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()

    return " ".join(list(map(str, vector)))


def get_parameter_combinations(params: List[float], n_sample: int=-1) -> Generator[List[float], None, None]:
    """
    Generate each time a combination of all the parameters (in order)
    such that each parameter is transformed like p_value * 10 ** (-x)
    where x is a value between 1 and 10. Moreover before generating
    all the combinations, all the parameters are scaled as values
    betwee 0 and 1 using `to_integral` function.

    It is possible to decide how many sample we would like the
    function to generate. The default value is -1, and in this case
    the number of generated samples is exactly p_len ** 10 // 2. 

    :param params:   The input vectors with all the parameters of the model
    :param n_sample: The number of samples that we want to generate
    :return: A generator that generate each time a sample
    """
    # Initialize the matrix with all the parameters already modified
    params = to_integral(params)
    np_param_matrix = np.array([params] * 10).T
    modifiers = np.array([[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]])
    path_matrix = np_param_matrix * modifiers

    # Initialize variables for the generation
    taken_combinations = dict()
    max_row_count = path_matrix.shape[0]
    current_sample_number = 0

    # Take the maximum number of possible combinations
    # that is exactly the number of columns power the
    # lenght of the combination. In practice we take the
    # 50% of the sample, otherwise there will be high
    if n_sample == -1:
        n_sample = path_matrix.shape[1] ** path_matrix.shape[0]
        n_sample = n_sample // 2

    while current_sample_number < n_sample:

        current_combination = []
        for current_row_index in range(0, max_row_count):
            current_row = path_matrix[current_row_index, :]
            chosen_value = np.random.choice(current_row)
            current_combination.append(chosen_value.item())
        
        current_combination_str = to_string(current_combination)
        if not current_combination_str in taken_combinations:
            taken_combinations[current_combination_str] = True
            current_sample_number += 1
            yield current_combination
    
    return


# -----------------------------------------------------------------------------
# Simulation Utilities
# -----------------------------------------------------------------------------

def setup_seed(seed: int=42) -> None:
    """
    Just setup the seed for the experiments

    :param seed: the seed
    :return:
    """
    # Setting up the seeds
    random.seed(seed)
    np.random.seed(seed)


def read_paths_file(paths_file: str) -> List[str]:
    """
    Read the content of the paths file. The file contains for each
    row the current path of a model inside a folder.

    :param paths_file: the absolute path to the file
    :return: a list with all the paths
    """
    paths = []
    with open(paths_file, mode="r", encoding="utf-8") as fhandle:
        while (line := fhandle.readline()):
            paths.append(line[:-1])
    
    return paths


def load_model(path_to_model: str) -> co.CModel:
    """
    Load the SBML model from a filename to a COPASI Model

    :param path_to_model: the absolute path to the SBML model file
    :return: a handle to the respective COPASI model
    """
    data_model: co.CDataModel = co.CRootContainer.addDatamodel()
    result = data_model.importSBML(path_to_model)

    # If loading failure
    if not result:
        print(f"Error: import SBML {path_to_model} failed")
        exit(1)

    model: co.CModel = data_model.getModel()
    return model


def print_species(model: co.CModel, foutput: Optional[str] = None) -> None:
    """
    Print all the species of the Model inside a file or on the std output.
    Whenever the output file is given, the content will be writte in CSV.
    
    :param model:   the COPASI Model
    :param foutput: an optional file where the write the output
    :return:
    """
    if not foutput:
        print("\n[*] Model Species Overview\n")

    # Get the total number of species inside the model
    num_species = model.getNumMetabs()

    # Define the dictionary that will construct the DataFrame
    pandas_df_dict = {
        "Specie"                : [], "Compartment": [], 
        "Initial Concentration" : [], "Expression" : [], 
        "Output"                : []
    }

    species, compartments, iconcentrations, expressions, outputs = [], [], [], [], []
    for nspecie in range(num_species):
        specie: co.CMetab = model.getMetabolite(nspecie)
        specie_name = specie.getObjectName()                    # Get the specie name
        compartment = specie.getCompartment().getObjectName()   # Get the compartment name
        initial_conc = specie.getInitialConcentration()         # Get the initial concentration
        expression = specie.getExpression()                     # Get the expression for that specie
        output = specie_name.endswith("_output")                # True if variable is for output, False otherwise

        species.append(specie_name)
        compartments.append(compartment)
        iconcentrations.append(initial_conc)
        expressions.append(expression)
        outputs.append(output)
    
    pandas_df_dict["Specie"] = species
    pandas_df_dict["Compartment"] = compartments
    pandas_df_dict["Initial Concentration"] = iconcentrations
    pandas_df_dict["Expression"] = expressions
    pandas_df_dict["Output"] = outputs

    df = pd.DataFrame(data=pandas_df_dict)

    # If the output file path is not given then
    # print on the standard output
    if not foutput:
        print(df)
        return

    # Otherwise write inside the file as CSV content
    df.to_csv(foutput)


def print_parameters(model: co.CModel, foutput: Optional[str]=None) -> None:
    """
    Print information about all the parameters of the model on file or std output.
    Whenever the output file is given, the content will be writte in CSV.

    :param model:   a handle to a COPASI model
    :param foutput: an optional file where the write the output
    :return:
    """
    if not foutput:
        print("\n[*] Model Parameters Overview\n")

    # Get the total number of parameters inside the model
    num_params = model.getNumModelValues()

    # Define the dictionary that will construct the DataFrame
    pandas_df_dict = {"Parameter": [], "Type": [], "Value": []}

    parameters, types, values = [], [], []
    for nparam in range(num_params):
        param: co.CModelValue = model.getModelValue(nparam)
        param_name = param.getObjectName()    # Get the name of the parameter
        param_value = param.getInitialValue() # Get the value of the parameter
        param_type = param.isFixed()          # True if the parameter is fixed, False otherwise

        parameters.append(param_name)
        types.append(param_type)
        values.append(param_value)

    pandas_df_dict["Parameter"] = parameters
    pandas_df_dict["Type"] = types
    pandas_df_dict["Value"] = values

    df = pd.DataFrame(data=pandas_df_dict)

    # If the output file path is not given then
    # print on the standard output
    if not foutput:
        print(df)
        return

    # Otherwise write inside the file as CSV content
    df.to_csv(foutput)


def handle_run_errors(task_id: str) -> None:
    """
    Print error messages from COPASI when the trajectory task failed.

    :param task_id: the ID of the task that call this function
    :return:
    """
    import sys
    sys.stderr.write(f"[*] <ERROR, Task ID:{task_id}> Running The time-course simulation failed.\n")

    # Check if there are additional messages
    if COPASI.CCopasiMessage.size() > 0:
        # Print the message on the standard error in chronological order
        sys.stderr.write(
            f"[*] <ERROR, Task ID: {task_id}> " + COPASI.CCopasiMessage.getAllMessageText(True)
        )


def load_report(report: str) -> pd.DataFrame:
    """
    Load the report file into a pandas DataFrame

    :param report: the path to the report file
    :return: a pandas DataFrame with the report
    """
    return pd.read_csv(report)


def plot(points: pd.DataFrame, vars: List[str]) -> None:
    """
    Plot one or more variables given a matrix of points

    :param points: a pandas DataFrame with the points
    :param vars: a list of variable names
    :return:
    """
    time_values = points["time"].values      # Take all times values
    vars_values = points.loc[:, vars].values # Take all variables values
    
    # Set some basic configuration for the plot
    plt.figure(figsize=[15.0, 8.0])
    plt.xlabel("Time")
    plt.ylabel("Species")

    # Iterate all the variables and plot
    for idx, var in enumerate(vars):
        plt.plot(time_values, vars_values[:, idx], label=var)

    plt.legend(loc="upper right")
    plt.show()


def get_mean_std(points: pd.DataFrame, vars: List[str]) -> pd.DataFrame:
    """
    Return a pandas dataFrame with only the mean and the standard
    deviation for all the variables specified in the input list.

    :param points: a pandas DataFrame with the points
    :param vars: a list of variable's name
    :return: a DataFrame with mean and std for all variables
    """
    description = points.describe()
    return description.loc[["mean", "std"], vars]


def normalize(points: pd.DataFrame, vars: List[str], ntype: str="statistical") -> pd.DataFrame:
    """
    Normalize each point (pointed by vars) in the input dataframe
    with the usual formula: z = (x - mean) / std. Then return
    the newly created and normalized dataframe. The applied 
    normalization depends on the input type: "statistical" means
    the one with mean and std. deviation; "classical" means between
    0 and 1, i.e., take the min value away and divide by the
    difference between the max and the min. 

    :param points: a pandas DataFrame with the points
    :param vars: a list of variable's name
    :param ntype: (optional) normalization type "statistical" or "classical"
    :return: the normalized dataframe
    """
    # First obtain the mean and the std deviation for each input variable
    description = points.describe()
    description = description.loc[:, vars]

    # Initialize the data for the new frame with the time column
    data = {"time" : points["time"].values.tolist()}

    # Iterate for each input variable and normalize them
    for var in vars:
        var_values = points.loc[:, var].values
        if ntype == "statistical":
            # Get the mean and the std for that variable
            mean = description.loc[["mean"], [var]].values.item()
            std  = description.loc[["std"],  [var]].values.item()

            # Normalize the variable values
            var_values = (var_values - mean) / std
        else:
            # Get the mininum and the maximum value
            min_value = description.loc[["min"], [var]].values.item()
            max_value = description.loc[["max"], [var]].values.item()

            # Normalize the variable values
            var_values = (var_values - min_value) / ((max_value - min_value) + sys.float_info.epsilon)

        data[var] = var_values.tolist()

    norm_df = pd.DataFrame(data, columns=["time"] + vars)
    return norm_df