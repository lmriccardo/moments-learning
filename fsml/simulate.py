from typing import List, Optional, Dict, Generator, Tuple
from dataclasses import dataclass
from multiprocessing import Pool
from datetime import datetime
import fsml.utils as utils
import os.path as opath
from basico import *
import COPASI as co
import pandas as pd
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Trajectory Task configurations and class
# -----------------------------------------------------------------------------

def generate_taskid(model_name: str, count: int) -> str:
    r"""
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
    r"""
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
    assert out_stype in ["Concentration", "ParticleNumber"], \
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

    # First add the FIXED parameters in the report
    num_parameters = model.getNumModelValues()
    for nparam in range(num_parameters):
        parameter: co.CModelValue = model.getModelValue(nparam)

        # Consider only FIXED quantities
        if parameter.getStatus() != co.CModelEntity.Status_FIXED:
            continue

        body.push_back(
            co.CRegisteredCommonName(
                parameter.getObject(
                    co.CCommonName(f"Reference=InitialValue")).getCN().getString()
            ))
    
        header.push_back(co.CRegisteredCommonName(co.CDataString(parameter.getSBMLId()).getCN().getString()))
        body.push_back(co.CRegisteredCommonName(report.getSeparator().getCN().getString()))
        header.push_back(co.CRegisteredCommonName(report.getSeparator().getCN().getString()))
    
    # Then add all the non-fixed species
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
                    co.CCommonName(f"Reference={out_stype}")).getCN().getString()
            ))
        
        # Add the corresponding ID to the header
        header.push_back(co.CRegisteredCommonName(co.CDataString(specie.getSBMLId() + "_specie").getCN().getString()))

        # After each entry we need a separator
        if nspecie != num_species - 1:
            body.push_back(co.CRegisteredCommonName(report.getSeparator().getCN().getString()))
            header.push_back(co.CRegisteredCommonName(report.getSeparator().getCN().getString()))
    
    return report


@dataclass(frozen=True)
class TaskConfiguration:
    r"""
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
    data_path : Tuple[str,str]
        A tuple with two elements, the first is the path for mean and std. output
        the second, instead, is the output path for the dense output.
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
    data_path            : Tuple[str,str]   # The path where to store the results


def generate_default_configuration() -> TaskConfiguration:
    r"""
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
    # First let's generate the data path
    mean_std_path = opath.join(os.getcwd(), "data/meanstd/")
    denseoutput_path = opath.join(os.getcwd(), "data/denseoutput/")

    # Try to create these two paths
    os.makedirs(mean_std_path, exist_ok=True)
    os.makedirs(denseoutput_path, exist_ok=True)

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
        report_fixed_species = True,
        data_path            = (mean_std_path, denseoutput_path)
    )


@dataclass()
class ParameterSamplerConfiguration:
    r""" Just a configuration class for the :class:`ParameterSampler` """
    total_number_of_sample : int    # Total number of sample to generate
    percentage_of_change   : float  # Percentage of change for each parameter
    total_sample_per_param : int    # Total number of sample per-parameter


def generate_default_param_sampler_configuration() -> ParameterSamplerConfiguration:
    r""" Generate a default configuration for the :class:`ParameterSampler` """
    return ParameterSamplerConfiguration(
        total_number_of_sample=10,
        percentage_of_change=50.0,
        total_sample_per_param=100
    )


class ParameterSampler:
    r"""A class for parameter sampling. 

    The goal is to generate each time a combination of all the parameters (in order)
    such that each parameter is transformed in according what's written 
    below. All the generated combinations are unique. A combination is generated 
    using a random walk on the matrix starting from the first row, picking an element
    at random and then going down in the remaining rows. 

    Given the list of parameters as input, let `N` be the size of the list, it returns
    a matrix of `N` rows times `nchange_per_param + 1` columns such that at each rows
    corresponds the vector of possible changes for the single parameter. That is, each
    row is a row vector of size `nchange_per_param + 1`, where the first element is 
    the original value of the corresponding parameter and all the other columns
    contains a random value. The random value is sampled from a Ball of radius exactly
    the `perc`% of the original value centered on the parameter. 

    Attributes
    ----------
    params : np.ndarray
        The vector with all the parameters
    conf : ParameterSamplerConfiguration
        The configuration for the sampler

    Methods
    -------
    generate(self) -> Generator[np.ndarray, None, None]
        Generate the actual sampling
    """
    def __init__(self, params: List[float], param_conf: ParameterSamplerConfiguration) -> None:
        """ Just the constructor """
        self.params = np.array(params).astype('float32') # The list with all the parameters value
        self.conf   = param_conf                         # The configuration for the sampler

    def _gen_parameter_matrix(self) -> np.ndarray:
        """ Generate the matrix for the sampling """
        # First compute the percentages of change for each parameter
        param_percent_vector = utils.compute_percent(self.params, self.conf.percentage_of_change)

        # Then create the matrix with all the possible samples for each parameter
        parameter_matrix = []
        for param, param_perc in zip(self.params, param_percent_vector):
            parameter_matrix.append(
                np.random.uniform((param - param_perc), 
                                  (param + param_perc), 
                                  size=(self.conf.total_sample_per_param)
                ).tolist())
            
        parameter_matrix = np.array(parameter_matrix).astype('float32')
        final_matrix = np.hstack((self.params.reshape(-1, 1), parameter_matrix))
        return final_matrix

    def generate(self) -> Generator[np.ndarray, None, None]:
        """ Generate the actual sampling """
        # Initialize the matrix with all the parameters already modified
        path_matrix = self._gen_parameter_matrix()

        # Initialize variables for the generation
        taken_combinations = dict()
        max_row_count = path_matrix.shape[0]
        current_sample_number = 0

        # Take the maximum number of possible combinations
        # that is exactly the number of columns power the
        # lenght of the combination. In practice we take the
        # 50% of the sample, otherwise there will be high
        n_sample = self.conf.total_number_of_sample
        if n_sample == -1:
            n_sample = path_matrix.shape[1] ** path_matrix.shape[0]
            n_sample = n_sample // 2

        while current_sample_number < n_sample:

            current_combination = []
            for current_row_index in range(0, max_row_count):
                current_row = path_matrix[current_row_index, :]
                chosen_value = np.random.choice(current_row)
                current_combination.append(chosen_value.item())
            
            current_combination_str = utils.to_string(current_combination)
            if not current_combination_str in taken_combinations:
                taken_combinations[current_combination_str] = True
                current_sample_number += 1
                yield current_combination
        
        return


class TrajectoryTask:
    r"""
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
    psampler_conf : ParameterSamplerConfiguration
        The configurator for the ParameterSampler
    mean_std_array : List[List[float]]
        The array that contains the output of the simulations
        such that each row is a simulation and the first N
        columns are the parameters while the last M columns
        are the mean and the standard deviation of the outputs

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

    def __init__(self, model_file : str,                          # A handle to the data model
                       log_dir    : str,                          # The path where to store the log file
                       output_dir : str,                          # The path where to store the output and the dense output
                       filename   : str,                          # The model filename (without the extension)
                       job        : int,                          # The Job Number
                       nsim       : int,                          # The total number of different simulations to run
                       param_conf : ParameterSamplerConfiguration # The configurator for the ParameterSampler
    ) -> None:
        r"""
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
        self.psampler_conf = param_conf
        self.psampler_conf.total_number_of_sample = self.num_simulations

        self.output_file = None
        self.res_file = None

        self.output_files = []
        self.res_files = []

        self.real_parameters_values = dict()
        self.modified_parameters_values = []

        self.mean_std_array = []
        self.dense_output_np = None
        self.dense_output_header = None

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

        # Create the parameter sampler
        parameter_sampler = ParameterSampler(parameter_values_list, self.psampler_conf)
        
        # Start with the generation
        for params_list in parameter_sampler.generate():
            yield dict(zip(parameter_names_list, params_list))
    
    def _change_parameter_values(self, param_dict: Dict[str, float]) -> None:
        """ Change the parameters of the model with the new values """
        # First take the model handler
        hmodel: co.CModel = self.datamodel.getModel()

        # Then iterate every parameter and set a new value
        number_of_parameters = hmodel.getNumModelValues()
        for nparam in range(number_of_parameters):
            current_parameter: co.CModelValue = hmodel.getModelValue(nparam)
            if not current_parameter.isFixed():
                continue

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
        r"""
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
        r"""
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
        r"""
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
    
    def __process_output(self, conf: TaskConfiguration, gen_denseoutput: bool) -> Tuple[List[str], List[str]]:
        """ Given a simulation result take the mean and std of that simulation """
        try:
            # First take the dense output from the output file
            dense_output = utils.load_report(self.output_file)
            variables = [x for x in dense_output.columns if x != "time" and x.endswith("_specie")]
            
            parameter_names = list(self.modified_parameters_values[-1].keys())

            if gen_denseoutput:
                # Take the dense output of the species and parameters
                self.dense_output_header = ["time"] + parameter_names + variables
                current_dense_output = utils.select_data(dense_output, self.dense_output_header)
                self.dense_output_np = self.dense_output_np.to_numpy() if self.dense_output_np is None \
                                else np.vstack((
                                    self.dense_output_np, current_dense_output.to_numpy()))
            
            # Compute the normalization and takes the mean and std for each variable
            class_normalization_variables = utils.normalize(dense_output, variables, ntype="classical")
            mean_std_variables = utils.get_mean_std(class_normalization_variables, variables)
            mean_std_variables_np = mean_std_variables.to_numpy().T.reshape(1, -1).tolist()
            mean_std_variables_names = []
            
            for variable in mean_std_variables.columns:
                mean_std_variables_names.append(f"mean_{variable}".upper())
                mean_std_variables_names.append(f"var_{variable}".upper())

            # Then take the current values used for the parameter and add them to the np array
            current_parameters = self.modified_parameters_values[-1]
            current_parameters_name = list(map(lambda x : x.lower(), parameter_names))
            current_parameters_value = list(current_parameters.values())

            # Finally save the final result
            self.mean_std_array.append(current_parameters_value + mean_std_variables_np[0])

            # Then eliminate the output file
            os.remove(self.output_file)
            
            # Return the name of the parameters and the name for the species
            return current_parameters_name, mean_std_variables_names
        
        except AssertionError as ae:
            ...

        return
    
    def run_task(self) -> bool:
        r"""
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

    def run(self, conf: TaskConfiguration, gen_denseoutput: bool=False) -> None:
        r"""
        Run the trajectory task

        :param conf: the Trajectory Task Configuration
        :param gen_denseoutput: True to genereate also the Denseoutput file
        :return:
        """
        mean_std_path, denseoutput_path = conf.data_path
        data_filename = bytes.fromhex(self.id.split("-")[0]).decode("ASCII")
        mean_std_filename = f"{data_filename}_MeanStd.csv"
        mean_std_filepath = opath.join(mean_std_path, mean_std_filename)

        # If it already exists then just remove it
        if opath.exists(mean_std_filepath):
            os.remove(mean_std_filepath)

        # 1. Print the basic information
        self.print_informations()

        # 2. Save the current parameters
        self.real_parameters_values = self._get_parameters()
        parameter_names, variable_names = None, None

        # 3. Start the new simulations with new parameters
        with tqdm(self._generate_new_parameters(), 
                  position=self.job, 
                  desc=f"Running Job for Model ID: {self.job}", 
                  leave=True,
                  total=self.num_simulations
        ) as progress_bar:
            for x in progress_bar:

                # 3.1. Change the model parameters and save the new model to file
                self._initialize_files()
                self._change_parameter_values(x)
                self.modified_parameters_values.append(x)

                # 3.2. Load the modified model into a new datamodel
                self.load_model()

                # 3.3. Setup the trajectory task
                self.setup_task(conf)

                # 3.4. Simulate
                progress_bar.set_postfix_str(f"Task ID: {self.id}")
                progress_bar.refresh()
                result = self.run_task()

                # If the simulation ended successfully then take the output
                if result:
                    parameter_names, variable_names = self.__process_output(conf, gen_denseoutput)

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

        # 4. Restore the original parameter value into the origina model
        self._change_parameter_values(self.real_parameters_values)

        # 5. End the simulations
        self.runned = True

        # 6. Save the result
        self.mean_std_array = np.array(self.mean_std_array)
        mean_std_df = pd.DataFrame(self.mean_std_array, columns=parameter_names + variable_names)
        mean_std_df.to_csv(mean_std_filepath, mode='a')

        if gen_denseoutput:
            denseoutput_filename = f"{data_filename}_DenseOutput.csv"
            denseoutput_filepath = opath.join(denseoutput_path, denseoutput_filename)

            denseoutput_df = pd.DataFrame(self.dense_output_np, columns=self.dense_output_header)
            denseoutput_df.to_csv(denseoutput_filepath)

    def get_used_parameters(self) -> List[Dict[str, float]]:
        """
        Return the list of all the parameter mapping used for each simulation

        :return: the parameter mapping
        """
        return self.modified_parameters_values


# -----------------------------------------------------------------------------
# Running simulations function
# -----------------------------------------------------------------------------

def run_one(
    model_path: str, log_dir: str, output_dir: str, data_dir: str, 
    job_id: int, nsim: int, gen_do: bool
) -> None:
    r"""
    Run `nsim` simulation for the single input model

    :param model_path: the absolute path to the SBML
    :param log_dir   : the absolute path to the log folder where to store the parameter and specie files
    :param output_dir: the absolute path to the output dir where to store the report and the result files
    :param data_dir  : the absolute path to the data folder where to store the output data
    :param job_id    : The ID of the job (just an integer)
    :param nsim      : the total number of simultations to run
    :param gen_do    : True if also the Dense Output should be generated
    :return:
    """
    # Get the filename of the model
    delimiter = "\\" if sys.platform == "win32" else "/"
    filename = model_path.split(delimiter)[-1].split("_")[0]

    # Setup and run the trajectory task
    task_conf = generate_default_configuration()
    param_conf = generate_default_param_sampler_configuration()
    ttask = TrajectoryTask(model_path, log_dir, output_dir, filename, job_id, nsim, param_conf)
    ttask.run(task_conf, gen_denseoutput=gen_do)

    # Save the results
    # generate_data_file(ttask, data_dir, gen_denseoutput=gen_do)


def run_simulations(
    paths_file: str, log_dir: str, output_dir: str,  data_dir: str, 
    nsim_per_model: int = 100, gen_denseoutput: bool=False
) -> None:
    r"""
    Run `nsim_per_model` simulations per each model belonging to the paths file
    where the paths file is a path that in each line contains the
    fully qualified path to the SBML of that model.

    :param paths_file    : the absolute path to the paths file
    :param log_dir   : the absolute path to the log folder where to store the parameter and specie files
    :param output_dir: the absolute path to the output dir where to store the report and the result files
    :param data_dir  : the absolute path to the data folder where to store the output data
    :param nsim_per_model: the number of simulations per model
    :param gen_denseoutput: True if also the Dense Output should be generated
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
                           nsim_per_model,
                           gen_denseoutput), 
                range(minc, maxc)
            ))

            pool.map(run_one, args)