import basico.biomodels as biomodels
from multiprocessing import Pool
from typing import Tuple, List
import os.path as opath
import libsbml
import math
import os


class Errors:
    FILE_NOT_EXISTS  = 1
    XML_READ_ERROR   = 2
    CONVERSION_ERROR = 3


###################################################################################################
################################## MODEL TRANSFORMATION FUNCTIONS #################################
###################################################################################################

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


def sbml_to_raterules(document: libsbml.SBMLDocument) -> libsbml.SBMLDocument:
    """
    Convert the SBML model by replacing all the reactions with
    the corresponding Rate Rules, i.e. ODE. In this way we also
    have that all the parameters local to reactions, now they
    become global to all the model. 

    :param document: a handler to the SBML document
    :return:
    """
    conv_properties = libsbml.ConversionProperties()
    conv_properties.addOption("replaceReactions")
    conversion_result = document.convert(conv_properties)
    if conversion_result != libsbml.LIBSBML_OPERATION_SUCCESS:
        print_errors(document, document.getNumErrors())
        exit(Errors.CONVERSION_ERROR)
    
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


def transform(sbmlfile: str, outputfile: str) -> None:
    """
    Load the model, transform the model and finally save the modified model.

    :param sbmlfile:   the absolute path to the SBML model to load
    :param outputfile: the absolute path to the file where to store the new SBML
    :return:
    """
    # Obtain the corresponding document
    document = load_sbml(sbmlfile)

    # Convert all reaction to rate rules
    document = sbml_to_raterules(document)

    # Get the handler to the SBML Model
    model = document.getModel()

    # Save the new model
    save_model(model, outputfile)


#####################################################################################################
################################## MODEL TRANSFORMATION DOWNLOADING #################################
#####################################################################################################


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


def transform_model(sbml_path: str) -> str:
    """
    Transform a SBML model using the transform function

    :param sbml_path: the path to the SBML model
    :return: the path where the transformed model has been saved
    """
    # Obtain the output path
    path, extension = sbml_path.split(".")
    output_path = f"{path}_output.{extension}"

    # Call the transform function to perform the transformation
    transform(sbml_path, output_path)

    return output_path


def convert_one(args: List[any]) -> str:
    """
    Convert one model speciefied by the model_id input parameter

    :param args: a List of arguments for the multiprocessing map function
        This parameter must contains only three elements:
            - prefix_path: the path where to save the model
            - model_id:    the ID of the model in the BioModels database
            - bvalue:      the normalization parameter

    :return: the path where the transformed model has been saved
    """
    # Extrapolate arguments
    prefix_path, model_id = args

    # Download the model
    model_path = download_model(prefix_path, model_id + 1)
    print(f"({model_id}) [*] Dowloaded file {model_path}")

    # Transform the model
    trans_model_path = transform_model(model_path)
    print(f"({model_id}) [*] Transformed model file {trans_model_path}")

    return trans_model_path


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


def convert_models(prefix_path: str, nmodels: int) -> List[str]:
    """
    Download and converts a bunch of models

    :param prefix_path: the path where to save the downloaded models
    :param nmodels:     the number of models to download and convert
    :return: a list containing all the paths to the transformed models
    """
    output_paths = []
    cpu_count = os.cpu_count()
    for (minc, maxc) in get_ranges(nmodels, cpu_count):
        with Pool(cpu_count) as pool:
            args = list(map(lambda x: (prefix_path, x), range(minc, maxc)))
            trans_model_paths = pool.map(convert_one, args)
            output_paths += trans_model_paths
    
    return output_paths


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


###################################################################################################
################################## MODEL FUNCTIONS FOR SIMULATIONS ################################
###################################################################################################

def to_unit(params: List[float]) -> List[float]:
    """
    Given a list of parameters convert each parameter from the 
    original value to a value between 1 and 9. That is, given
    a parameter value x we obtain z = x / (10 ** int(log10(x))).

    :param params: the list with all the parameters value
    :return: the new list of parameters
    """
    scale = lambda x: x / (10 ** int(math.log10(x)))
    new_params = list(map(scale, params))
    return new_params


def get_list_possible_params(params: List[float]) -> List[List[float]]:
    """
    :param params: the list with all the parameters value
    :return: 
    """


def main() -> None:
    prefix_path = opath.join(os.getcwd(), "tests")
    nmodels = 1

    paths = convert_models(prefix_path, nmodels)
    write_paths(paths, opath.join(os.getcwd(), "data/paths.txt"))
    remove_original(paths)


if __name__ == "__main__":
    main()