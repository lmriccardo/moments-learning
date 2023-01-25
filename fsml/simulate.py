from typing import List, Optional, Dict, Generator
from dataclasses import dataclass
from multiprocessing import Pool
from datetime import datetime
import fsml.utils as utils
import os.path as opath
from basico import *
import COPASI as co
import pandas as pd


# -----------------------------------------------------------------------------
# Trajectory Task configurations and class
# -----------------------------------------------------------------------------

def generate_taskid(model_name: str, count: int) -> str:
    """
    Generate a new task ID. The ID is in the following format:
    <hour><min><sec>-<number>-<filename>, where all the component
    are in Hexadecimal form. <number> is an integer that is 
    sequentially incremented each time a new task is generated. 

    :param model_name: the name of the model
    :param count: just an integer
    :return: the ID
    """
    local_time = datetime.datetime.now()
    hour = hex(local_time.hour)[2:]
    minute = hex(local_time.minute)[2:]
    seconds = hex(local_time.second)[2:]

    count += 1

    hex_modelname = "".join(list(map(lambda x: hex(ord(x))[2:], model_name)))

    return f"{hex_modelname}-{hex(count)[2:]}-{hour}{minute}{seconds}"


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
    report.setSeparator(co.CCopasiReportSeparator(","))

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


def generate_default_configuration() -> TaskConfiguration:
    """
    Generate a default configuration for the Trajectory Task.
    The default configuration is the following:

        - step size = 0.01
        - initial time = 0.0
        - simulation horizon = 100.0
        - generate time series = True
        - automatic step size = False
        - output event = False
        - absolute tolerance 1.0e-9
        - relative tolerance 1.0e-9
        - output also fixed species = False
        - output concentration or amount ? Concentration

    :return: a new TaskConfiguration object already configured 
    """
    return TaskConfiguration(
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
    output_file : str
        the file path where to store the final resulsts of the simulation
    res_file : str
        the file path where to store all the dense output for the simulation
    filename : str
        the filename of the model (without the extension)
    num_simulations : int
        The total number of simulations
    real_parameters_value : Dict[str, float]
        A dictionary containing the original values for the parameters of the model
    modified_parameters_values : List[Dict[str, float]]
        A List containing mapping from parameters name and parameter values
    runned : bool
        If the trajectory task has been runned or not

    Methods
    -------
    get_model(self) -> co.CModel:
        Return the model handler of the current datamodel

    def load_model(self) -> None:
        Load a new model from file
    
    print_informations(self) -> None
        Print all the useful information about the input model

    setup_task(self, conf: TaskConfiguration) -> None
        Setup different options for the trajectory task.
    
    run_task(self) -> bool
        Run the trajectory task and return True if Ok, False otherwise

    print_result(self) -> None
        Print the result of the simulation to the output file
    """

    def __init__(self, model_file : str, # A handle to the data model
                       log_dir    : str, # The path where to store the log file
                       output_dir : str, # The path where to store the output and the dense output
                       filename   : str, # The model filename (without the extension)
                       job        : int, # The Job Number
                       nsim       : int  # The total number of different simulations to run
    ) -> None:
        """
        :param datamodel : a handle to the COPASI Data model
        :param log_dir   : The path where to store the log files
        :param output_dir: The path where to store the output and the dense output
        :param filename  : The model filename (without the extension)
        :param job       : The Job ID of the task
        :param nsim      : total number of different simulations to run
        """
        self.model_path      = model_file
        self.datamodel       = None
        self.count           = 0
        self.id              = generate_taskid(filename, count=self.count)
        self.trajectory_task = None
        
        self.log_path = log_dir
        self.output_path = output_dir
        self.filename = filename
        self.job = job
        self.num_simulations = nsim

        self.output_file = None
        self.res_file = None

        self.output_files = []
        self.res_files = []

        self.real_parameters_values = dict()
        self.modified_parameters_values = []

        self.runned = False

        self.load_model()

    def _initialize_files(self):
        """ Initialize all the filenames required for the output """
        self.output_file = opath.join(self.output_path, f"{self.id}_report.txt")
        self.res_file = opath.join(self.output_path, f"{self.id}_res.txt")

        self.output_files.append(self.output_file)
        self.res_files.append(self.res_file)

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

    def _get_parameters(self) -> Dict[str, float]:
        """ Obtain the parameters of the model from the CSV file """
        # Obtain the parameters from the CSV
        params_filename = f"{self.filename}_parameters.csv"
        params_path = opath.join(self.log_path, params_filename)
        parameters_df = pd.read_csv(params_path)

        # Obtain a dictionary with keys names and values parameter values
        parameters_dictionary = dict()
        parameters_df_dict = parameters_df.to_dict()
        for idx, p_name in parameters_df_dict["Parameter"].items():
            parameters_dictionary[p_name] = parameters_df_dict["Value"][idx]
        
        return parameters_dictionary

    def _generate_new_parameters(self) -> Generator[Dict[str, float], None, None]:
        """ Generate new parameters values for the model """
        # First we need to get all the parameters value
        parameter_dictionary = self._get_parameters()

        # Extract from the dictionary the actual values in a list
        parameter_values_list = list(parameter_dictionary.values())

        # Extract also the parameter names in a list
        parameter_names_list = list(parameter_dictionary.keys())
        
        # Start with the generation
        for params_list in utils.get_parameter_combinations(parameter_values_list, n_sample=self.num_simulations):
            yield dict(zip(parameter_names_list, params_list))
    
    def _change_parameter_values(self, param_dict: Dict[str, float]) -> None:
        """ Change the parameters of the model with the new values """
        # First take the model handler
        hmodel: co.CModel = self.datamodel.getModel()

        # Then iterate every parameter and set a new value
        number_of_parameters = hmodel.getNumModelValues()
        for nparam in range(number_of_parameters):
            current_parameter: co.CModelValue = hmodel.getModelValue(nparam)
            new_parameter_value = param_dict[current_parameter.getObjectName()]
            current_parameter.setInitialValue(new_parameter_value)
            
        self.datamodel: co.CDataModel = hmodel.getObjectDataModel()
        self.datamodel.exportSBML(self.model_path, overwriteFile=True, sbmlLevel=3, sbmlVersion=2)

    def get_model(self) -> co.CModel:
        """ Return the model handler of the current datamodel """
        return self.datamodel.getModel()
    
    def load_model(self) -> None:
        """ Load a new model from file """
        model = load_model(self.model_path)
        self.datamodel = model.getObjectDataModel()

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
        utils.print_species(model_handler, species_path)
        utils.print_parameters(model_handler, params_path)

    def setup_task(self, conf: TaskConfiguration) -> None:
        """
        Setup different options for the trajectory task.

        :param conf: the configuration to be applied
        :return:
        """
        self.trajectory_task = self.datamodel.getTask("Time-Course")

        # Check that the returned task is a Trajectory Task
        assert isinstance(self.trajectory_task, COPASI.CTrajectoryTask), \
            f"<ERROR, Task ID: {self.id}> Not a trajectory Task"

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
                utils.handle_run_errors(self.id)
                return False

            # If no error occured then just return True
            return True
        except Exception:
            utils.handle_run_errors(self.id)
            return False

    def run(self, conf: TaskConfiguration) -> None:
        """
        Run the trajectory task

        :param conf: the Trajectory Task Configuration
        :return:
        """
        # 1. Print the basic information
        self.print_informations()

        # 2. Save the current parameters
        self.real_parameters_values = self._get_parameters()

        print(f"[*] Starting Job: {self.job}")

        # 3. Start the new simulations with new parameters
        for x in self._generate_new_parameters():

            # 3.1. Change the model parameters and save the new model to file
            self._initialize_files()
            self._change_parameter_values(x)
            self.modified_parameters_values.append(x)

            # 3.2. Load the modified model into a new datamodel
            self.load_model()

            # 3.3. Setup the trajectory task
            self.setup_task(conf)

            # 3.4. Simulate
            print(f"[*] Running Job {self.job} Task ID: {self.id}")
            result = self.run_task()

            # 3.5. Generate a new Task ID and initialize the files
            # for the new incoming simulation
            self.count += 1
            self.id = generate_taskid(self.filename, count=self.count)

            if not result:
                # Remove res and report file from the list
                self.output_files.pop()
                self.res_files.pop()
                self.modified_parameters_values.pop()
                os.remove(self.output_file)
                continue
            
            # 3.6. Print the final results
            self.print_results()

        # 4. Restore the original parameter value into the origina model
        self._change_parameter_values(self.real_parameters_values)

        # 5. End the simulations
        self.runned = True

    def get_used_parameters(self) -> List[Dict[str, float]]:
        """
        Return the list of all the parameter mapping used for each simulation

        :return: the parameter mapping
        """
        return self.modified_parameters_values


# -----------------------------------------------------------------------------
# Generation Results file function
# -----------------------------------------------------------------------------


def generate_data_file(trajectory_task: TrajectoryTask, data_path: Optional[str]=None) -> None:
    """
    Takes as input the Trajectory Task that has been runned
    and generate a new file CSV in the data folder such that
    each row is a simulation and columns are divided as follow:
    first N columns are the initial values of the parameters, 
    the following 2 * M columns are the mean and the std.
    deviation computed for each species throgh the entire simulation.

    :param trajectory_task: A handle to a TrajectoryTask Object
    :param data_path      : The full qualified path of the data folder
    :return:  
    """
    # Check that the simulations has been runned
    assert trajectory_task.runned, \
        f"<ERROR, Job {trajectory_task.job}> No simulation has been runned"

    # Check if the data folder is either None or it does exists
    if not data_path or not opath.exists(opath.abspath(data_path)):

        try:
            # If does not exists or it is not then create a new data folder
            data_path = opath.join(os.getcwd(), "data/simulations")
            os.mkdir(data_path)
        except FileExistsError:
            ... # Maybe the given path wasn't a full path

    data_path = opath.abspath(data_path)

    # Obtain the parameter list
    parameters_list = trajectory_task.get_used_parameters()

    # Load the results for each simulation as dictionaries of (normalized) values
    species_mean_std: Dict[str, List[float]] = dict()
    for output_file in trajectory_task.output_files:

        try:
            # Load the report and produce a DenseOutput DataFrame
            dense_output = utils.load_report(output_file)
            variables = [x for x in dense_output.columns if x != "time"]

            # Compute the normalization and takes the mean and std for each variable
            class_normalization_variables = utils.normalize(dense_output, variables, ntype="classical")
            mean_std_variables = utils.get_mean_std(class_normalization_variables, variables)
            mean_std_variables_dict = mean_std_variables.to_dict()

            # Initialize the dictionary of species and fill it with the values
            for variable in variables:
                mean_var = f"mean_{variable}".upper()
                std_var  = f"std_{variable}".upper()

                if not mean_var in species_mean_std:
                    species_mean_std[mean_var] = []
                    species_mean_std[std_var] = []
            
                mean_var_value = mean_std_variables_dict[variable]["mean"]
                std_var_value  = mean_std_variables_dict[variable]["std"]

                species_mean_std[mean_var].append(mean_var_value)
                species_mean_std[std_var].append(std_var_value)

        except AssertionError as ae:
            ...

    # Then we need to flatten the parameter list of dictionaries
    # into a dictionary of parameters list
    parameters_dictionary = { p.lower() : [] for p in parameters_list[0].keys() }
    for parameter_dict in parameters_list:
        for parameter, value in parameter_dict.items():
            parameters_dictionary[parameter.lower()].append(value)
        
    # Then we need to merge the two dictionaries and create the DataFrame
    df_dict = parameters_dictionary
    df_dict.update(species_mean_std)
    data_df = pd.DataFrame(data=df_dict)

    # Craft the name of the data file that will contains the data
    delimiter = "\\" if sys.platform == "win32" else "/"
    data_filename = output_file.split("-")[0].split(delimiter)[-1] + ".csv"
    data_file = opath.join(data_path, data_filename)
    data_df.to_csv(data_file)

    # Remove the report and res file
    # for report_file, res_file in zip(trajectory_task.output_files, trajectory_task.res_files):
    #     if opath.exists(report_file): os.remove(report_file)
    #     if opath.exists(res_file): os.remove(res_file)


# -----------------------------------------------------------------------------
# Running simulations function
# -----------------------------------------------------------------------------

def run_one(
    model_path: str, log_dir: str, output_dir: str, data_dir: str, job_id: int, nsim: int
) -> None:
    """
    Run `nsim` simulation for the single input model

    :param model_path: the absolute path to the SBML
    :param log_dir   : the absolute path to the log folder where to store the parameter and specie files
    :param output_dir: the absolute path to the output dir where to store the report and the result files
    :param data_dir  : the absolute path to the data folder where to store the output data
    :param job_id    : The ID of the job (just an integer)
    :param nsim      : the total number of simultations to run
    :return:
    """
    # Get the filename of the model
    delimiter = "\\" if sys.platform == "win32" else "/"
    filename = model_path.split(delimiter)[-1].split("_")[0]

    # Setup and run the trajectory task
    task_conf = generate_default_configuration()
    ttask = TrajectoryTask(model_path, log_dir, output_dir, filename, job_id, nsim)
    ttask.run(task_conf)

    # Save the results
    generate_data_file(ttask, data_dir)


def run_simulations(
    paths_file: str, log_dir: str, output_dir: str,  data_dir: str, nsim_per_model: int = 100
) -> None:
    """
    Run `nsim_per_model` simulations per each model belonging to the paths file
    where the paths file is a path that in each line contains the
    fully qualified path to the SBML of that model.

    :param paths_file    : the absolute path to the paths file
    :param log_dir   : the absolute path to the log folder where to store the parameter and specie files
    :param output_dir: the absolute path to the output dir where to store the report and the result files
    :param data_dir  : the absolute path to the data folder where to store the output data
    :param nsim_per_model: the number of simulations per model
    :return
    """
    path_list = utils.read_paths_file(paths_file)
    cpu_count = os.cpu_count()
    for (minc, maxc) in utils.get_ranges(len(path_list), cpu_count):
        with Pool(cpu_count) as pool:
            args = list(
                map(
                lambda x: (path_list[x], 
                           log_dir, 
                           output_dir, 
                           data_dir, 
                           x, 
                           nsim_per_model), 
                range(minc, maxc)
            ))

            pool.map(run_one, args)