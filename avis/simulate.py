from typing import List, Optional
from dataclasses import dataclass
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
    local_time = datetime.datetime.now()
    hour = hex(local_time.hour)[2:]
    minute = hex(local_time.minute)[2:]
    seconds = hex(local_time.second)[2:]
    
    global CURRENT_TASK_ID
    CURRENT_TASK_ID += 1

    hex_modelname = "".join(list(map(lambda x: hex(ord(x))[2:], model_name)))

    return f"{hour}{minute}{seconds}-{hex(CURRENT_TASK_ID)[2:]}-{hex_modelname}"


def create_report(datamodel     : co.CDataModel, 
                  task_id       : str, 
                  out_stype     : str, 
                  fixed_species : bool=False) -> co.CReportDefinition:
    """
    Function to create a report definition which will contain all the
    information (i.e. time step and species values) from a time-course
    simulation. The report will have N + 1 columns, where N is equal to
    the number of species and the +1 is given by the "time" column, and
    a number of row that is equal to the horizon of the simulation times
    the size of each step (in the case a fixed time step has been chosen).

    :param datamodel: a handle to a COPASI DataModel
    :param task_id:   the string representing the ID of the task
    :param out_stype: (possible values are "Concetration" (or "concentration")
                      and "Amount" (or "amount")) if we want to consider the 
                      concentration or the amount value for each specie.
    :param fixed_species: if in the output we want to put also FIXED value species.
    :return: the report definition
    """
    assert out_stype.capitalize() in ["Concentration", "Amount"], \
        f"<ERROR, Task ID: {task_id}> In Create Report given output Specie Type {out_stype}.\n" + \
         "\tInstead specify one among: Concentration (or concentration), Amount (or amount)"

    # Obtain an handle to the model
    model: co.CModel = datamodel.getModel()

    # Create a report with the correct filename and all the species against time
    reports: co.CReportDefinitionVector = datamodel.getReportDefinitionList()

    # Create a report definition object
    report: co.CReportDefinition = reports.createReportDefinition(
        f"Report {task_id}",  "Output for Time-Course Task"
    )

    # Set the task type for the report definition to timecourse
    report.setTaskType(co.CTaskEnum.Task_timeCourse)

    # We don't want a table
    report.setIsTable(False)

    # The entry of the output should be separated by ", "
    report.setSeparator(co.CCopasiReportSeparator(", "))

    # We need a handle to the header and the body
    # the header will display the ids of the metabolites
    # and "time" for the first column. The body will contain
    # the actual time course data.
    header: co.ReportItemVector = report.getHeaderAddr()
    body: co.ReportItemVector = report.getBodyAddr()
    body.push_back(
        co.CRegisteredCommonName(
            co.CCommonName(
                model.getCN().getString() + ",Reference=Time").getString()
        ))
    body.push_back(co.CRegisteredCommonName(report.getSeparator().getCN().getString()))
    header.push_back(co.CRegisteredCommonName(co.CDataString("time").getCN().getString()))
    header.push_back(co.CRegisteredCommonName(report.getSeparator().getCN().getString()))
    num_species = model.getNumMetabs()
    for nspecie in range(num_species):
        specie: co.CMetab = model.getMetabolite(nspecie)
        
        # In case fixed_species is True we don't want
        # to consider also FIXED metabolites in the output
        if fixed_species and specie.getStatus() == co.CModelEntity.Status_FIXED:
            continue

        # Set if we want concentration or amount
        body.push_back(
            co.CRegisteredCommonName(
                specie.getObject(
                    co.CCommonName(f"Reference={out_stype.capitalize()}")).getCN().getString()
            ))
        
        # Add the corresponding ID to the header
        header.push_back(co.CRegisteredCommonName(co.CDataString(specie.getSBMLId()).getCN().getString()))

        # After each entry we need a separator
        if nspecie != num_species - 1:
            body.push_back(co.CRegisteredCommonName(report.getSeparator().getCN().getString()))
            header.push_back(co.CRegisteredCommonName(report.getSeparator().getCN().getString()))
    
    return report


@dataclass(frozen=True)
class TaskConfiguration:
    """
    This Python dataclass is used the represent a COPASI trajectory task configuration.
    A Trajectory task configuration is used to set the value for the simulation
    step number, the simulation initial time, the horizon, absolute and relative tolerances ...
    This class is given as input to the `obj:TrajectoryTask.setup_task` function.

    Attributes
    ----------
    step_size : float or None
        the step size of the simulation
    initial_time : float
        The initial time of the simulation
    sim_horizon : float
        The horizon of the simulation
    get_time_series : bool
        Tell the task (or problem) to generate the time series
    automatic_step_size : bool
        Tell the task to use automatic step size or not. This should be
        set to False whenever we set `step_size` to None. Otherwise, 
        we can safely set True.
    output_event : bool
        Tell COPASI if we want additional points for event assignments
    abs_tolerance : float
        the absolute tolerance
    rel_tolerance : float
        the relative tolerance
    report_out_stype : str
        The output type of the species in the report (concentration or amount)
    report_fixed_species : bool
        If consider in the report also FIXED value species
    """
    step_size            : Optional[float]  # The step size of the simulation
    initial_time         : float            # The initial time of the simulation
    sim_horizon          : float            # The horizon of the simulation
    gen_time_series      : bool             # Tell the problem to generate the time series
    automatic_step_size  : bool             # Use automatic step size or not
    output_event         : bool             # Tell COPASI if we want additional points for event assignment
    abs_tolerance        : float            # Absolute tolerance parameter
    rel_tolerance        : float            # Relative tolerance parameter
    report_out_stype     : str              # The output type of the species in the report (concentration or amount)
    report_fixed_species : bool             # If consider in the report also FIXED value species


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
    print_informations(self) -> None
        Print all the useful information about the input model

    setup_task(self, conf: TaskConfiguration) -> None
        Setup different options for the trajectory task.
    
    run_task(self) -> bool
        Run the trajectory task and return True if Ok, False otherwise

    print_result(self) -> None
        Print the result of the simulation to the output file
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

        self.output_file = opath.join(self.output_path, f"{self.id}_report.txt")
        self.res_file = opath.join(self.output_path, f"{self.id}_res.txt")

        self._create_directories() # Create the log and output folder if don't exist

    def _create_directories(self) -> None:
        """ Create input directories if they do not exists """
        # Log folder checking
        try:
            os.mkdir(self.log_path)
        except FileExistsError:
            pass
        
        try:
            os.mkdir(self.output_path)
        except FileExistsError:
            pass

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

    def setup_task(self, conf: TaskConfiguration) -> None:
        """
        Setup different options for the trajectory task.

        :param conf: the configuration to be applied
        :return:
        """
        # Run a deterministic time course
        self.trajectory_task.setMethodType(co.CTaskEnum.Method_deterministic)

        # The task will run when the module will be saved
        self.trajectory_task.setScheduled(True)

        # Create the report and set report options
        report = create_report(self.datamodel, self.id, conf.report_out_stype, conf.report_fixed_species)
        self.trajectory_task.getReport().setReportDefinition(report)
        self.trajectory_task.getReport().setTarget(self.output_file)
        self.trajectory_task.getReport().setAppend(False)

        # Get the problem for the task to set some parameters
        problem: co.CTrajectoryProblem = self.trajectory_task.getProblem()
        problem.setStepSize(conf.step_size)    # Set the step size
        problem.setDuration(conf.sim_horizon)  # Set the horizon of the simulation
        problem.setTimeSeriesRequested(True)   # Request all the time series
        problem.setAutomaticStepSize(conf.automatic_step_size)  # Set automatic step size
        problem.setOutputEvent(False)  # we don't want more output points for event assignments

        # set some parameters for the LSODA method through the method
        method = self.trajectory_task.getMethod()
        abs_param = method.getParameter("Absolute Tolerance")
        rel_param = method.getParameter("Relative Tolerance")
        abs_param.setValue(conf.abs_tolerance)
        rel_param.setValue(conf.rel_tolerance)

        # Set the initial time of the simulation
        self.datamodel.getModel().setInitialTime(conf.initial_time)

    def run_task(self) -> bool:
        """
        Run a Time-Course simulation with COPASI and returns TRUE
        If no error occurred, otherwise it returns False.

        :param trajectory_task: a handle to the Time-Course Task
        :return: True if no errors, False otherwise
        """
        try:
            # Run the trajectory task
            result = self.trajectory_task.process(True)

            # Check if some error occurred
            if not result:
                handle_run_errors(self.id)
                return False

            # If no error occured then just return True
            return True
        except Exception:
            handle_run_errors(self.id)
            return False

    def print_results(self) -> None:
        """
        This method prints the final result of the simulation to
        the output file. The output file is identified by the task ID
        with an _res suffix. The file will be stored in the output
        folder given as input to the constructor of the class. In the
        result file will be visible: the total number of steps of the
        simulation, the total number of variables for each step and
        the final state, i.e., the final values for each species.
        """
        # Get the time series object
        time_series = self.trajectory_task.getTimeSeries()

        # Craft the output file in the output folder
        open_mode = "x" if not opath.exists(self.res_file) else "w"

        # Open the file and write the final content
        with open(self.res_file, mode=open_mode, encoding="utf-8") as hfile:
            hfile.write("N. Steps Time Series: {0} steps\n".format(time_series.getRecordedSteps()))
            hfile.write("N. Variable Each Step: {0} vars\n".format(time_series.getNumVariables()))
            hfile.write("\nThe Final State is:\n")

            # Get the number of variables and iterates
            num_vars = time_series.getNumVariables()
            last_index = time_series.getRecordedSteps() - 1
            for nvar in range(0, num_vars):
                # Here we get the particle numbers (at least for species)
                hfile.write("    {0}: {1}\n".format(time_series.getTitle(nvar), time_series.getData(last_index, nvar)))
        
        return

    def run(self, conf: TaskConfiguration) -> None:
        """
        Run the trajectory task

        :param conf: the Trajectory Task Configuration
        :return:
        """
        # 1. Print the basic information
        self.print_informations()

        # 2. Setup the trajectory task
        self.setup_task(conf)

        # 3. Simulate
        print(f"[*] Running Task ID: {self.id}")
        result = self.run_task()

        assert result, \
            f"<ERROR, Task ID: {self.id}> Simulation Failed"
        
        # 4. Print the final results
        self.print_results()


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        model_path = "C:\\Users\\ricca\\Desktop\\Projects\\Avis\\tests\\BIOMD00001_output.xml"
        log_dir = "C:\\Users\\ricca\\Desktop\\Projects\\Avis\\log\\"
        output_dir = "C:\\Users\\ricca\\Desktop\\Projects\\Avis\\runs\\"
    else:
        model_path = "/Users/yorunoomo/Desktop/Projects/moments-learning/tests/BIOMD00001_output.xml"
        log_dir = "/Users/yorunoomo/Desktop/Projects/moments-learning/log/"
        output_dir = "/Users/yorunoomo/Desktop/Projects/moments-learning/runs/"

    task_conf = TaskConfiguration(
        step_size            = 0.01,
        initial_time         = 0.0,
        sim_horizon          = 100.0,
        gen_time_series      = True,
        automatic_step_size  = False,
        output_event         = False,
        abs_tolerance        = 1.0e-09,
        rel_tolerance        = 1.0e-09,
        report_out_stype     = "Concentration",
        report_fixed_species = True
    )

    model = load_model(model_path)
    datamodel = model.getObjectDataModel()
    ttask = TrajectoryTask(datamodel, log_dir, output_dir, "BIOMD00001")
    ttask.run(task_conf)
