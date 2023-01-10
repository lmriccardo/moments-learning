from typing import Dict, Tuple, List
import basico.biomodels as biomodels
from multiprocessing import Pool
import os.path as opath
import libsbml
import os


class Errors:
    FILE_NOT_EXISTS = 1
    XML_READ_ERROR  = 2


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


def print_species(species: Dict[str, Tuple[str]]) -> None:
    """
    Print all the species parsed from the model

    :param species: the dictionary of all species
    """
    # Obtain the name with the maximum length
    max_len_name = max(map(lambda x: len(x), species.keys()))

    # Iterate and print
    for idx, (specie_name, (specie_id, _)) in enumerate(species.items()):
        spaces = " " * ((max_len_name - len(specie_name)) + 1)
        print("N. %02d) Specie Name: %s%s| Specie Id: %s" % (
            idx + 1, specie_name, spaces, specie_id
        ))


def get_species(model: libsbml.Model) -> Dict[str, Tuple[str, str]]:
    """
    Obtain all the species names as long as their IDs

    :param model: A handler to the SBML Model
    :return: a dictionary with keys specie's name and value specie's ID
    """
    species = dict()

    # Obtain the total number of species in the model
    num_species = model.getNumSpecies()
    for i in range(0, num_species):

        # Take the specie at position i
        current_specie: libsbml.Species = model.getSpecies(i)

        # Get the name and the ID and fill the dictionary
        species[current_specie.getName()] = (current_specie.getId(), current_specie.getCompartment())

    return species


def add_output_species(model: libsbml.Model, species: Dict[str, Tuple[str]]) -> List[libsbml.Species]:
    """
    For each specie inside a SBML model, add a corresponding output specie
    which output value will be the normalization of all the values that
    the original specie get through all the simulation process.

    Each new specie will belong to the same compartment of the original specie,
    will have constant attribute False and boundaryCondition attribute True.

    :param model:   a handler to the SBML Model object
    :param species: the dictionary containing all the species of the model
    :return: a list containing all the new SBML Species created
    """
    new_species_list = []
    for idx, (specie_name, (specie_id, specie_compartment)) in enumerate(species.items()):
        # Set new name, new id and new compartment for the new specie
        new_name = specie_name + "_output"
        new_id   = specie_id + "_output"
        new_comp = specie_compartment

        # Create a new empty specie with level 3 version 2
        new_specie = model.createSpecies()

        # Set the new attributes
        new_specie.setName(new_name)
        new_specie.setId(new_id)
        new_specie.setCompartment(new_comp)
        new_specie.setConstant(False)
        new_specie.setBoundaryCondition(True)

        new_species_list.append(new_specie)

    return new_species_list


def add_bparameter(model: libsbml.Model, bvalue: float) -> libsbml.Parameter:
    """
    Add a new constant parameter called "normalization_bvalue" to the Model

    :param model:  a handle to the SBML model
    :param bvalue: the value of the new parameter
    :return: a handle to the newly created parameter
    """
    bparameter: libsbml.Parameter = model.createParameter()
    bparameter.setName("normalization_bvalue")
    bparameter.setId("bparameter")
    bparameter.setValue(bvalue)
    bparameter.setConstant(True)

    return bparameter

    
def add_normalization_function(model: libsbml.Model) -> libsbml.FunctionDefinition:
    """
    Add a new function for the model that computes the normalization
    of the state (specie) values at each step of the simulation. This
    is used to compute the output value for the new output variable
    (specie) previously added to the model. This function takes as
    input a specie, and a parameter and compute

            Y = (1 + (1 - exp(-b * z)) / (1 + exp(-b * z))) / 2

    where 'p' is the parameter and 'z' is the state.

    :param model: a handle to the SBML model
    :return: the handle to tne newly created function
    """
    nfunction: libsbml.FunctionDefinition = model.createFunctionDefinition()
    nfunction.setId("nfunction")
    nfunction.setName("State Normalization Function")

    # Create the Math for the function definition
    function_def_mathml = "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">" + \
                          "    <lambda>"                                        + \
                          "        <bvar><ci>z</ci></bvar>"                     + \
                          "        <bvar><ci>b</ci></bvar>"                     + \
                          "        <apply>"                                     + \
                          "            <divide/>"                               + \
                          "            <apply>"                                 + \
                          "                <plus/>"                             + \
                          "                <cn>1</cn>"                          + \
                          "                <apply>"                             + \
                          "                    <divide/>"                       + \
                          "                    <apply>"                         + \
                          "                        <minus/>"                    + \
                          "                        <cn>1</cn>"                  + \
                          "                        <apply>"                     + \
                          "                            <exp/>"                  + \
                          "                            <apply>"                 + \
                          "                                <times/>"            + \
                          "                                <apply>"             + \
                          "                                    <minus/>"        + \
                          "                                    <ci>b</ci>"      + \
                          "                                </apply>"            + \
                          "                                <ci>z</ci>"          + \
                          "                            </apply>"                + \
                          "                        </apply>"                    + \
                          "                    </apply>"                        + \
                          "                    <apply>"                         + \
                          "                        <plus/>"                     + \
                          "                        <cn>1</cn>"                  + \
                          "                        <apply>"                     + \
                          "                            <exp/>"                  + \
                          "                            <apply>"                 + \
                          "                                <times/>"            + \
                          "                                <apply>"             + \
                          "                                    <minus/>"        + \
                          "                                    <ci>b</ci>"      + \
                          "                                </apply>"            + \
                          "                                <ci>z</ci>"          + \
                          "                            </apply>"                + \
                          "                        </apply>"                    + \
                          "                    </apply>"                        + \
                          "                </apply>"                            + \
                          "            </apply>"                                + \
                          "            <cn>2</cn>"                              + \
                          "        </apply>"                                    + \
                          "    </lambda>"                                       + \
                          "</math>"

    # Obtain the ASTNode that corresponds to the mathml formula
    mathml_astnode = libsbml.readMathMLFromString(function_def_mathml)

    # Set the formula for the function
    nfunction.setMath(mathml_astnode)

    return nfunction


def add_rules(model     : libsbml.Model, 
              species   : Dict[str, Tuple[str, str]], 
              ospecies  : List[libsbml.Species], 
              bparam    : libsbml.Parameter,
              nfunction : libsbml.FunctionDefinition) -> None:
    """
    For each species add an assignment rule in the model that associate the
    corresponding new output specie with the normalization of the value which
    that specie has in any time of the simulation. That is, for each specie
    add a rule like: specie_output = norm_function(specie, bparam).

    :param model:     a handle to the SBML Model
    :param species:   a dictionary containing all the original specie of the model
    :param ospecies:  a list of all the new output species
    :param bparam:    the normalization parameter
    :param nfunction: a handle to the normalization function definition
    :return:
    """
    for idx, (_, (specie_id, _)) in enumerate(species.items()):
        # Obtain the corresponding output specie
        ospecie = ospecies[idx]

        # Add a new assignment rule in the model
        assignment_rule: libsbml.AssignmentRule = model.createAssignmentRule()
        assignment_rule.setVariable(ospecie.getId())
        assignment_rule.setId(f"{ospecie.getId()}_{specie_id}_assrule")
        assignment_rule.setName(f"Normalization {specie_id} to {ospecie.getId()}")

        # Create and assign a new ASTNode
        specie_node = libsbml.ASTNode(libsbml.AST_NAME)
        bparam_node = libsbml.ASTNode(libsbml.AST_NAME)
        function_node = libsbml.ASTNode(libsbml.AST_FUNCTION)

        specie_node.setName(specie_id)
        bparam_node.setName(bparam.getId())
        function_node.setName(nfunction.getId())

        function_node.addChild(specie_node)
        function_node.addChild(bparam_node)

        assignment_rule.setMath(function_node)


def save_model(model: libsbml.Model, path: str) -> None:
    """
    Write a SBML Model inside a file

    :param model: a handle to the SBML model
    :param path:  a string representing the output path
    """
    writer = libsbml.SBMLWriter()
    document = model.getSBMLDocument()
    writer.writeSBMLToFile(document, path)


def transform(sbmlfile: str, outputfile: str, bvalue: float) -> None:
    """
    Load the model, transform the model and finally save the modified model.
    The transformation involves: adding new output species, which will have
    the normalized value for each respective specie through the entire
    simulation; adding a new parameter, the normalization parameter; adding
    a new funciton, the normalization function that takes as input a specie
    and the normalization parameter and compute the normalized value; adding
    for each specie an assignment rule that maps the current value of that
    specie with the output specie. 

    :param sbmlfile:   the absolute path to the SBML model to load
    :param outputfile: the absolute path to the file where to store the new SBML
    :param bvalue:     the input value for the normalization parameter
    :return:
    """
    # Obtain the corresponding document
    document = load_sbml(sbmlfile)
    model = document.getModel()

    # Obtain all the species and add output species to the model
    species = get_species(model)
    output_species = add_output_species(model, species)

    # Add the new parameter
    bparameter = add_bparameter(model, bvalue)

    # Add the new function to the model
    nfunction = add_normalization_function(model)

    # Add new rules
    add_rules(model, species, output_species, bparameter, nfunction)

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


def transform_model(sbml_path: str, bvalue: float) -> str:
    """
    Transform a SBML model using the transform function

    :param sbml_path: the path to the SBML model
    :param bvalue:    the normalization parameter used for the transformation
    :return: the path where the transformed model has been saved
    """
    # Obtain the output path
    path, extension = sbml_path.split(".")
    output_path = f"{path}_output.{extension}"

    # Call the transform function to perform the transformation
    transform(sbml_path, output_path, bvalue)

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
    prefix_path, model_id, bvalue = args

    # Download the model
    model_path = download_model(prefix_path, model_id + 1)
    print(f"({model_id}) [*] Dowloaded file {model_path}")

    # Transform the model
    trans_model_path = transform_model(model_path, bvalue)
    print(f"({model_id}) [*] Transformed model file {trans_model_path}")

    return trans_model_path


def convert_models(prefix_path: str, bvalue: float, nmodels: int) -> List[str]:
    """
    Download and converts a bunch of models

    :param prefix_path: the path where to save the downloaded models
    :param bvalue:      the normalization parameter used for the transformation
    :param nmodels:     the number of models to download and convert
    :return: a list containing all the paths to the transformed models
    """
    output_paths = []
    cpu_count = os.cpu_count()
    for nmodel in range(0, nmodels, cpu_count):
        with Pool(cpu_count) as pool:
            args = list(map(lambda x: (prefix_path, x, bvalue), range(nmodel, nmodel + cpu_count)))
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


def main() -> None:
    prefix_path = opath.join(os.getcwd(), "tests")
    nmodels = 1
    bvalue = 0.01

    paths = convert_models(prefix_path, bvalue, nmodels)
    write_paths(paths, opath.join(os.getcwd(), "data/paths.txt"))
    remove_original(paths)


if __name__ == "__main__":
    main()