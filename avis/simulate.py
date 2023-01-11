from typing import List, Optional
from datetime import datetime
import os.path as opath
from basico import *
import COPASI as co
import pandas as pd


#####################################################################################################
################################## UTILITY FUNCTION FOR SIMULATION ##################################
#####################################################################################################


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

    # Print model overview
    print_model(data_model)

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
    if foutput:
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
    if foutput:
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


#####################################################################################################
################################## TRAJECTORY TASK CLASS DEFINITION #################################
#####################################################################################################


CURRENT_TASK_ID = 0


def generate_taskid(model_name: str) -> str:
    """
    Generate a new task ID. The ID is in the following format:
    <hour><min><sec>-<number>-<filename>, where all the component
    are in Hexadecimal form. <number> is an integer that is 
    sequentially incremented each time a new task is generated. 

    :return: the ID
    """
    local_time = datetime.now()
    hour = hex(local_time.hour)[2:]
    minute = hex(local_time.minute)[2:]
    seconds = hex(local_time.second)[2:]
    
    global CURRENT_TASK_ID
    CURRENT_TASK_ID += 1

    hex_modelname = "".join(list(map(lambda x: hex(ord(x))[2:], model_name)))

    return f"{hour}{minute}{seconds}-{hex(CURRENT_TASK_ID)[2:]}-{hex_modelname}"


class TrajectoryTask:
    """
    A class used to represent a COPASI Trajectory Task. Each taks is independent
    from the others and can be identified by an appropriate Task ID. With this class
    the user can setup a Trajectory Task, initialize all the parameters like
    the horizon of the simulation, the initial time step, if automatic step size or not,
    absolute and relative tolerances and more other options. Finally, once the setup
    is done, the user can run the simulation and store the result in appropriate folders.

    Attributes
    ----------
    datamodel : COPASI.CDataModel
        a handle to the COPASI data model that contains the actual SBML model
    id : str
        the identifier of the task (unique and generated using generate_taskid function)
    trajectory_task : COPASI.CTrajectoryTask
        the actual trajectory task returned by the data model
    log_path : str
        the path where to store the file containing the log of the simulation
    output_path : str
        the path where to store the file containing the results of the simulation
    filename : str
        the filename of the model (without the extension)

    Methods
    -------
    print_informations()
        Print all the useful information about the input model

    """

    def __init__(self, datamodel  : COPASI.CDataModel, # A handle to the data model
                       log_dir    : str,               # The path where to store the log file
                       output_dir : str,               # The path where to store the output and the dense output
                       filename   : str                # The model filename (without the extension)
    ) -> None:
        """
        :param datamodel : a handle to the COPASI Data model
        :param log_dir   : The path where to store the log files
        :param output_dir: The path where to store the output and the dense output
        :param filename  : The model filename (without the extension)
        """
        self.datamodel       = datamodel
        self.id              = generate_taskid(filename)
        self.trajectory_task = self.datamodel.getTask("Time-Course")

        # Check that the returned task is a Trajectory Task
        assert isinstance(self.trajectory_task, COPASI.CTrajectoryTask), \
            f"<ERROR, Task ID: {self.id}> Not a trajectory Task"
        
        self.log_path = log_dir
        self.output_path = output_dir
        self.filename = filename

    def print_informations(self) -> None:
        """
        Print all the useful information about the input model
        into two files `<log_path>\<filename>_species.csv` and
        `<log_path>\<filename>_parameters.csv`. The stored
        informations involve all the species of the model and 
        all its paremeters.
        """
        # Create the filename to store the species of the model
        species_filename = f"{self.filename}_species.csv"
        species_path = opath.join(self.log_path, species_filename)

        # Create the filename to store the parameters of the model
        params_filename = f"{self.filename}_parameters.csv"
        params_path = opath.join(self.log_path, params_filename)

        # Prints these two information into the respective file
        model_handler = self.datamodel.getModel()
        print_species(model_handler, species_path)
        print_parameters(model_handler, params_path)


def run_trajectory_task(trajectory_task: COPASI.CTrajectoryTask, task_id: str) -> bool:
    """
    Run a Time-Course simulation with COPASI and returns TRUE
    If no error occurred, otherwise it returns False.

    :param trajectory_task: a handle to the Time-Course Task
    :return: True if no errors, False otherwise
    """
    try:
        # Run the trajectory task
        result = trajectory_task.process(True)

        # Check if some error occurred
        if not result: 
            handle_run_errors(task_id)

        # If no error occured then just return True
        return True
    except Exception:
        handle_run_errors(task_id)

    finally:
        return False


if __name__ == "__main__":
    model = load_model("C:\\Users\\ricca\\Desktop\\Projects\\Avis\\tests\\BIOMD00001_output.xml")
    print_species(model, "C:\\Users\\ricca\\Desktop\\Projects\\Avis\\tests\\BIOMD00001_species.csv")
    print_parameters(model, "C:\\Users\\ricca\\Desktop\\Projects\\Avis\\tests\\BIOMD00001_parameters.csv")